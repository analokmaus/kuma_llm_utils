import openai
import os
import re
import asyncio
from pathlib import Path
from PIL import Image
from collections import defaultdict

from ..utils import create_batches, load_image_for_llm
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
    """A client class for interacting with OpenAI's API.
    This class implements the AbstractLLMEngine interface to provide OpenAI-specific
    functionality for generating text completions. It handles API key management,
    usage limits, and both synchronous and asynchronous text generation.
    Args:
        api_key (str, optional): OpenAI API key. If not provided, will attempt to retrieve
            from environment variable 'OPENAI_API_KEY'.
        usage_limits (dict, optional): Dictionary defining usage limits for different models.
            Defaults to openai_default_limits.
    Attributes:
        api_key (str): The provided API key.
        client (openai.OpenAI): The OpenAI client instance.
        usage_limits (defaultdict): Dictionary containing LimitManager instances for each model.
    Methods:
        generate(message, generation_params): 
            Asynchronously generates a completion for a single message.
        generate_parallel(messages, generation_params):
            Asynchronously generates completions for multiple messages in parallel.
    Example:
        >>> client = OpenAIClient(api_key="your-api-key")
        >>> response = await client.generate(
        ...     message=[{"role": "user", "content": "Hello"}],
        ...     generation_params={"model": "gpt-3.5-turbo"}
        ... )
    """
    
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
             'input_token': response.usage.prompt_tokens,
             'output_token': response.usage.completion_tokens,
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
    """
    A worker class for handling OpenAI API interactions with templated prompts and batched generation capabilities.
    This class implements the AbstractLLMWorker interface specifically for OpenAI's API, providing
    functionality for single and parallel text generation with customizable prompts and system messages.
    Parameters
    ----------
    engine : OpenAIClient
        The OpenAI client instance used for API communication.
    prompt_template : str, optional
        Template string for formatting prompts with placeholders (default: '').
    prompt_default_fields : dict, optional
        Default values for prompt template fields (default: {}).
    prompt_required_fields : list[str], optional
        List of required field names that must be present in the prompt template (default: []).
    system_prompt : str, optional
        System-level prompt to be prepended to all conversations (default: None).
    generation_params : dict, optional
        Parameters for text generation (default: openai_default_params).
    Attributes
    ----------
    is_async : bool
        Indicates that this worker operates asynchronously (True).
    Methods
    -------
    generate(inputs: list[dict]) -> str
        Asynchronously generates a response for a single input.
    generate_parallel(inputs: list[list], batch_size: int = 4) -> list
        Asynchronously generates responses for multiple inputs in parallel.
    generate_parallel_streaming(inputs: list[list], batch_size: int = 4) -> Generator
        Asynchronously generates responses for multiple inputs in parallel with streaming.
    generate_batched(inputs: list[list], batch_size: int = 4)
        Not implemented.
    generate_batched_streaming(inputs: list[list], batch_size: int = 4)
        Not implemented.
    Raises
    ------
    ValueError
        If required fields are missing from the prompt template or if an unsupported role is provided.
    """

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

    def _fill_template(self, **kwargs):
        if self.template == '':
            if len(kwargs) == 1:
                return list(kwargs.values())[0]
            else:
                return '\n' .join([f'{k}:{v}' for k, v in kwargs.items()])
        else:
            prompt_fields = self.prompt_default_fields.copy()
            prompt_fields.update(kwargs)
            return self.template.format(**prompt_fields)

    def _get_prompt(self, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._fill_template(**kwargs),
                    }
                ]
            }
        ]
        return messages

    def _parse_inputs(self, inputs: list[dict]):
        parsed_inputs = []
        for input_dict in inputs:
            if 'role' not in input_dict.keys():
                input_dict['role'] = 'user'
            role = input_dict.pop('role')
            if role == 'user':
                parsed_input = self._get_prompt(**input_dict)
            elif role in ['assistant', 'model']:
                parsed_input = [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": input_dict['text']}],
                }]
            else:
                raise ValueError(f'{self.name} role {role} is not supported.')
            parsed_inputs.extend(parsed_input)
        if self.system_prompt is not None:
            parsed_inputs.insert(0, {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            })
        return parsed_inputs
    
    def _parse_logprobs(self, logprobs):
        logprobs_json = []
        for logprob in logprobs.content:
            logprobs_json.append(dict(
                token=logprob.token,
                logprob=logprob.logprob
            ))
        return logprobs_json

    def _check_template(self):
        template_fields = set(re.findall(r'{([^{}]*)}', self.template))
        missing_fields = set(self.prompt_required_fields) - template_fields
        if len(missing_fields) > 0:
            raise ValueError(f'{self.name} fields {missing_fields} are required in prompt template.')

    async def generate(self, inputs: dict | list[dict], return_logprobs: bool = False):
        if isinstance(inputs, dict):
            inputs = [inputs]
        response = await self.engine.generate(self._parse_inputs(inputs), self.generation_params)
        decoded_output = response.choices[0].message.content
        if return_logprobs:
            return decoded_output, self._parse_logprobs(response.choices[0].logprobs)
        else:
            return decoded_output

    async def generate_parallel(self, inputs: list[list], batch_size: int = 4):
        messages = [self._parse_inputs(inputs_item) for inputs_item in inputs]
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

    async def generate_parallel_streaming(self, inputs: list[list], batch_size: int = 4):
        messages = [self._parse_inputs(inputs_item) for inputs_item in inputs]
        messages_batches = create_batches(messages, batch_size)

        for batch_i, batch in enumerate(messages_batches):
            self.logger.info(f'{self.name} processing batch {batch_i+1}/{len(messages_batches)}')
            responses = await self.engine.generate_parallel(batch, self.generation_params)

            decoded_output_batch = []
            for response in responses:
                decoded_output_batch.append(response.choices[0].message.content)

            yield decoded_output_batch

    async def generate_batched(self, inputs: list[list], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')

    async def generate_batched_streaming(self, inputs: list[list], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')


class OpenAIVisionWorker(OpenAIWorker):
    """
    A worker class for handling vision-based tasks using OpenAI's GPT-4 Vision API.

    This class extends OpenAIWorker to provide functionality for processing and analyzing images
    along with text prompts. It handles image resizing, format conversion, and proper message
    formatting for the OpenAI Vision API.

    Parameters
    ----------
    engine : OpenAIClient
        The OpenAI client instance for making API calls
    prompt_template : str, optional
        Template string for formatting prompts
    prompt_default_fields : dict, optional
        Default fields to use in the prompt template
    prompt_required_fields : list[str], optional
        List of required fields that must be provided in the prompt
    system_prompt : str, optional
        System-level prompt to guide the model's behavior
    generation_params : dict, optional
        Parameters for text generation (defaults to openai_default_params)
    high_image_quality : bool, optional
        If True, uses high quality image processing, otherwise low (default: False)
    image_max_size : int, optional
        Maximum size in pixels for image dimension (default: 1568)

    Attributes
    ----------
    image_quality : str
        Quality setting for image processing ('high' or 'low')
    image_max_size : int
        Maximum size in pixels for image dimension

    Methods
    -------
    _resize_image(image: Image.Image)
        Resizes an image while maintaining aspect ratio
    _process_image(image: str | Path | Image.Image)
        Processes different image input types into API-compatible format
    _get_prompt(**kwargs)
        Generates the formatted prompt with image for API request

    Notes
    -----
    - Supports image inputs as file paths, URLs, or PIL Image objects
    - Automatically resizes images that exceed the maximum dimension while preserving aspect ratio
    - Converts images to base64 encoding when necessary
    """

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

    def _process_image(self, image: str | Path | Image.Image):
        base64_image, media_type = load_image_for_llm(image, self.image_max_size)
        image_url = f"data:{media_type};base64,{base64_image}"
        return dict(
            type='image_url',
            image_url=dict(url=image_url, detail=self.image_quality))

    def _get_prompt(self, **kwargs):
        image_contents = []
        image_keys = [k for k in kwargs.keys() if re.match(r'^image(\d)*$', k)]
        for k in image_keys:
            image = kwargs.pop(k)
            image_contents.append(self._process_image(image))
        messages = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": self._fill_template(**kwargs),
                }] + image_contents
            }
        ]
        return messages
