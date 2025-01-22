import anthropic
import os
import re
import asyncio
import httpx
import base64
from pathlib import Path
from PIL import Image
from collections import defaultdict

from ..utils import create_batches, pil_to_base64, is_url
from .abstract import AbstractLLMEngine, AbstractLLMWorker
from .utils import LimitManager


anthropic_default_params = dict(
    model='claude-3-5-sonnet-20241022',
    max_tokens=1024,
    temperature=0.,
    top_p=1.
)

anthropic_default_limits = {
    'claude-3-5-sonnet-latest': [
        {'request': 900, 'input_token': 72000, 'output_token': 14400, 'reset_cycle': 60}],
    'claude-3-5-sonnet-20241022': [
        {'request': 900, 'input_token': 72000, 'output_token': 14400, 'reset_cycle': 60}],
    'claude-3-5-sonnet-20240620': [
        {'request': 900, 'input_token': 72000, 'output_token': 14400, 'reset_cycle': 60}],
    'claude-3-5-haiku-latest': [
        {'request': 900, 'input_token': 90000, 'output_token': 18000, 'reset_cycle': 60}],
    'claude-3-5-haiku-20241022': [
        {'request': 900, 'input_token': 90000, 'output_token': 18000, 'reset_cycle': 60}],
    'claude-3-opus-latest': [
        {'request': 900, 'input_token': 36000, 'output_token': 7200, 'reset_cycle': 60}],
    'claude-3-opus-20240229': [
        {'request': 900, 'input_token': 36000, 'output_token': 7200, 'reset_cycle': 60}],
}


class AnthropicClient(AbstractLLMEngine):
    def __init__(
            self,
            api_key: str = None,
            usage_limits: dict = anthropic_default_limits):
        super().__init__()
        self.api_key = api_key
        self.__api_key = self._retrieve_api_key()
        self.client = anthropic.Anthropic(api_key=self.__api_key)
        self.usage_limits = defaultdict(list)
        for model_name, limits_list in usage_limits.items():
            for limits in limits_list:
                self.usage_limits[model_name].append(LimitManager(limits=limits, logger=self.logger))

    async def _call_time_manager(self):
        for _, manager_list in self.usage_limits.items():
            for manager in manager_list:
                await manager.check()

    def _update_counter(self, model_name: str, usage: dict):
        for manager in self.usage_limits[model_name]:
            manager.add(usage)

    def _retrieve_api_key(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.api_key is not None:
            api_key = self.api_key
            self.logger.warning(f'{self.name} the API key in the environment variable has been overridden by the argument.')
        if api_key is None:
            self.logger.critical(f'{self.name} API key not found.')
        return api_key

    async def _async_generate(self, message, generation_params):
        await self._call_time_manager()
        response = await asyncio.to_thread(
            self.client.messages.create,
            messages=message,
            **generation_params
        )
        self._update_counter(
            generation_params['model'],
            {'request': 1, 'input_token': response.usage.input_tokens,
             'output_token': response.usage.output_tokens})
        return response

    async def generate(self, message, generation_params):
        return await asyncio.create_task(self._async_generate(message, generation_params))

    async def generate_parallel(self, messages, generation_params):
        tasks = [asyncio.create_task(
            self._async_generate(message, generation_params)) for message in messages]
        responses = [await task for task in tasks]
        return responses


class AnthropicWorker(AbstractLLMWorker):

    def __init__(
            self,
            engine: AnthropicClient,
            prompt_template: str = '',
            prompt_default_fields: dict = {},
            prompt_required_fields: list[str] = [],
            system_prompt: str = None,
            generation_params: dict = anthropic_default_params
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
        decoded_output = response.content[0].text
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
                decoded_output_batch.append(response.content[0].text)

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
                decoded_output_batch.append(response.content[0].text)
                self.time_manager['input_token'] += response.usage.input_tokens
                self.time_manager['output_token'] += response.usage.output_tokens
                self.time_manager['num_requests'] += 1

            yield decoded_output_batch

    async def generate_batched(self, inputs: list[dict], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')

    async def generate_batched_streaming(self, inputs: list[dict], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')


class AnthropicVisionWorker(AnthropicWorker):

    def __init__(
            self,
            engine: AnthropicClient,
            prompt_template: str = '',
            prompt_default_fields: dict = {},
            prompt_required_fields: list[str] = [],
            system_prompt: str = None,
            generation_params: dict = anthropic_default_params,
            image_max_size: int = 1568
    ):
        super().__init__(
            engine,
            prompt_template,
            prompt_default_fields,
            prompt_required_fields,
            system_prompt,
            generation_params
        )
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
                base64_image = base64.standard_b64encode(httpx.get(image).content).decode("utf-8")
                media_type = f'image/{Path(image).suffix}'
            else:
                pil_img = Image.open(image)
                pil_img = self._resize_image(pil_img)
                base64_image = pil_to_base64(pil_img, 'png')
                media_type = 'image/png'

        elif isinstance(image, Image.Image):
            image = self._resize_image(image)
            base64_image = pil_to_base64(image, 'png')
            media_type = 'image/png'
        return dict(type='base64', media_type=media_type, data=base64_image)

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
                        "type": "image",
                        "source": image_content,
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
