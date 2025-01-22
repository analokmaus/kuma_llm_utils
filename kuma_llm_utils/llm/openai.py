import openai
import os
import re
import asyncio
from pathlib import Path
from PIL import Image
from collections import defaultdict

from ..utils import create_batches, pil_to_base64, is_url
from .abstract import AbstractLLMEngine, AbstractLLMWorker
from .utils import LimitManager


openai_default_params = dict(
    model='gpt-4o-2024-08-06',
    max_tokens=1024,
    temperature=0.,
    top_p=1.
)

openai_default_limits = {
    'gpt-4o': [{'request': 450, 'token': 27000, 'reset_cycle': 60}],
    'gpt-4o-mini': [
        {'request': 450, 'token': 180000, 'reset_cycle': 60},
        {'request': 9000, 'token': None, 'reset_cycle': 3600*24}
    ],
    'o1-mini': [{'request': 450, 'token': 180000, 'reset_cycle': 60}],
}


class OpenAIClient(AbstractLLMEngine):
    def __init__(
            self,
            api_key: str = None,
            usage_limits: dict = openai_default_limits):
        super().__init__()
        self.api_key = api_key
        self.__api_key = self._retrieve_api_key()
        self.client = openai.OpenAI(api_key=self.__api_key)
        self.usage_limits = defaultdict(list)
        for model_name, limits_list in usage_limits.items():
            for limits in limits_list:
                self.usage_limits[model_name].append(LimitManager(limits=limits, logger=self.logger))

    async def _call_time_manager(self):
        for _, manager_list in self.usage_limits.items():
            for manager in manager_list:
                await manager.check()

    def _update_counter(self, model_name: str, usage: dict):
        self.logger.info(f'{self.name} {model_name} {usage}')
        for manager in self.usage_limits[model_name]:
            manager.add(usage)

    def _retrieve_api_key(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key is not None:
            api_key = self.api_key
            self.logger.warning(f'{self.name} the API key in the environment variable has been overridden by the argument.')
        if api_key is None:
            self.logger.critical(f'{self.name} API key not found.')
        return api_key

    async def _async_generate(self, message, generation_params):
        await self._call_time_manager()
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            messages=message,
            **generation_params
        )
        self._update_counter(
            generation_params['model'],
            {'request': 1,
             'token': response.usage.prompt_tokens + response.usage.completion_tokens})
        return response

    async def generate(self, message, generation_params):
        return await asyncio.create_task(self._async_generate(message, generation_params))

    async def generate_parallel(self, messages, generation_params):
        tasks = [asyncio.create_task(
            self._async_generate(message, generation_params)) for message in messages]
        responses = [await task for task in tasks]
        return responses


class OpenAIWorker(AbstractLLMWorker):

    def __init__(
            self,
            engine: OpenAIClient,
            prompt_template: str = '',
            prompt_default_fields: dict = {},
            prompt_required_fields: list[str] = [],
            system_prompt: str = None,
            generation_params: dict = openai_default_params
    ):
        super().__init__()
        self.engine = engine
        self.template = prompt_template
        self.prompt_default_fields = prompt_default_fields
        self.prompt_required_fields = prompt_required_fields
        self.system_prompt = system_prompt
        self.generation_params = generation_params
        self._check_template()
        self.is_async = True

    def _get_prompt(self, **kwargs):
        prompt_fields = self.prompt_default_fields.copy()
        prompt_fields.update(kwargs)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.template.format(**prompt_fields),
                    }
                ]
            }
        ]
        if self.system_prompt is not None:
            messages.insert(0, {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            })
        return messages

    def _check_template(self):
        template_fields = set(re.findall(r'{([^{}]*)}', self.template))
        missing_fields = set(self.prompt_required_fields) - template_fields
        if len(missing_fields) > 0:
            raise ValueError(f'{self.name} fields {missing_fields} are required in prompt template.')

    async def generate(self, **kwargs):
        response = await self.engine.generate(self._get_prompt(**kwargs), self.generation_params)
        decoded_output = response.choices[0].message.content
        return decoded_output

    async def generate_parallel(self, inputs: list[dict], batch_size: int = 4):
        messages = [self._get_prompt(**kwargs) for kwargs in inputs]
        messages_batches = create_batches(messages, batch_size)

        decoded_outputs = []
        for batch_i, batch in enumerate(messages_batches):
            self.logger.info(f'{self.name} processing batch {batch_i+1}/{len(messages_batches)}')
            responses = await self.engine.generate_parallel(batch, self.generation_params)

            decoded_output_batch = []
            for response in responses:
                decoded_output_batch.append(response.choices[0].message.content)

            decoded_outputs.extend(decoded_output_batch)

        return decoded_outputs

    async def generate_parallel_streaming(self, inputs: list[dict], batch_size: int = 4):
        messages = [self._get_prompt(**kwargs) for kwargs in inputs]
        messages_batches = create_batches(messages, batch_size)

        for batch_i, batch in enumerate(messages_batches):
            self.logger.info(f'{self.name} processing batch {batch_i+1}/{len(messages_batches)}')
            responses = await self.engine.generate_parallel(batch, self.generation_params)

            decoded_output_batch = []
            for response in responses:
                decoded_output_batch.append(response.choices[0].message.content)

            yield decoded_output_batch

    async def generate_batched(self, inputs: list[dict], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')

    async def generate_batched_streaming(self, inputs: list[dict], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')


class OpenAIVisionWorker(OpenAIWorker):

    def __init__(
            self,
            engine: OpenAIClient,
            prompt_template: str = '',
            prompt_default_fields: dict = {},
            prompt_required_fields: list[str] = [],
            system_prompt: str = None,
            generation_params: dict = openai_default_params,
            high_image_quality: bool = False,
            image_max_size: int = 1568
    ):
        super().__init__(
            engine,
            prompt_template,
            prompt_default_fields,
            prompt_required_fields,
            system_prompt,
            generation_params,
        )
        self.image_quality = "high" if high_image_quality else "low"
        self.image_max_size = image_max_size

    def _resize_image(self, image: Image.Image):
        width, height = image.size
        max_size = self.image_max_size
        if max(width, height) < max_size:
            return image
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
        image_resized = image.resize((new_width, new_height), Image.BICUBIC)
        self.logger.info(f'{self.name} image resized ({width}, {height}) -> ({new_width}, {new_height})')
        return image_resized

    def _process_image(self, image: str | Path | Image.Image):
        if isinstance(image, str | Path):
            if is_url(image):
                img_url = image
            else:
                pil_img = Image.open(image)
                pil_img = self._resize_image(pil_img)
                base64_image = pil_to_base64(pil_img, 'png')
                img_url = f"data:image/png;base64,{base64_image}"
        elif isinstance(image, Image.Image):
            image = self._resize_image(image)
            base64_image = pil_to_base64(image, 'png')
            img_url = f"data:image/png;base64,{base64_image}"
        return dict(url=img_url, detail=self.image_quality)

    def _get_prompt(self, **kwargs):
        if 'image' in kwargs.keys():
            image = kwargs.pop('image')
            image_content = self._process_image(image)
        else:
            raise ValueError('image must be attached.')
        prompt_fields = self.prompt_default_fields.copy()
        prompt_fields.update(kwargs)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.template.format(**prompt_fields),
                    },
                    {
                        "type": "image_url",
                        "image_url": image_content,
                    }
                ]
            }
        ]
        if self.system_prompt is not None:
            messages.insert(0, {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            })
        return messages
