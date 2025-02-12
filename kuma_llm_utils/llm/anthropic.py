import anthropic
import os
import re
import asyncio
from pathlib import Path
from PIL import Image
from collections import defaultdict

from ..utils import create_batches, load_image_for_llm
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
    """A client class for interacting with the Anthropic API.
    This class provides an interface to generate responses using Anthropic's language models,
    with built-in usage limits management and async support.
    Args:
        api_key (str, optional): The Anthropic API key. If not provided, will attempt to retrieve from environment variables.
        usage_limits (dict, optional): Dictionary defining usage limits for different models. Defaults to anthropic_default_limits.
    Attributes:
        api_key (str): The provided API key.
        client (anthropic.Anthropic): The Anthropic client instance.
        usage_limits (defaultdict): Dictionary storing LimitManager instances for each model.
    Methods:
        generate(message, generation_params):
            Asynchronously generates a single response.
        generate_parallel(messages, generation_params):
            Asynchronously generates multiple responses in parallel.
    Example:
        >>> client = AnthropicClient(api_key="your-api-key")
        >>> response = await client.generate(message, {"model": "claude-2"})
    Note:
        - Requires valid Anthropic API credentials
        - Implements rate limiting and usage tracking
        - Inherits from AbstractLLMEngine
    """

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
        self.logger.info(f'{self.name} {model_name} {usage}')
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
        responses = await asyncio.gather(*tasks)
        return responses


class AnthropicWorker(AbstractLLMWorker):
    """
    A worker class for handling interactions with Anthropic's API.
    This class implements the AbstractLLMWorker interface for Anthropic's language models,
    providing methods for generating text responses both individually and in parallel.
    Args:
        engine (AnthropicClient): The client instance for interacting with Anthropic's API.
        prompt_template (str, optional): Template string for formatting prompts. Defaults to ''.
        prompt_default_fields (dict, optional): Default fields to use in the prompt template. Defaults to {}.
        prompt_required_fields (list[str], optional): List of required fields in the prompt template. Defaults to [].
        system_prompt (str, optional): System prompt to prepend to all interactions. Defaults to None.
        generation_params (dict, optional): Parameters for text generation. Defaults to anthropic_default_params.
    Attributes:
        engine (AnthropicClient): The Anthropic client instance.
        template (str): The prompt template string.
        prompt_default_fields (dict): Default fields for the prompt template.
        prompt_required_fields (list[str]): Required fields for the prompt template.
        system_prompt (str): The system prompt.
        generation_params (dict): Parameters for text generation.
        is_async (bool): Indicates that this worker operates asynchronously.
    Methods:
        generate(inputs: list[dict]): Generate a single response from the model.
        generate_parallel(inputs: list[list], batch_size: int = 4): Generate multiple responses in parallel.
        generate_parallel_streaming(inputs: list[list], batch_size: int = 4): Stream multiple responses in parallel.
        generate_batched(inputs: list[list], batch_size: int = 4): Not implemented.
        generate_batched_streaming(inputs: list[list], batch_size: int = 4): Not implemented.
    Raises:
        ValueError: If required fields are missing in the prompt template or if an unsupported role is provided.
    """


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
        for _input_dict in inputs:
            input_dict = _input_dict.copy()
            if 'role' not in input_dict.keys():
                input_dict['role'] = 'user'
            role = input_dict.pop('role')
            if role == 'user':
                parsed_input = self._get_prompt(**input_dict)
            elif role in ['assistant', 'model', 'system']:
                assert 'content' in input_dict.keys() or 'text' in input_dict.keys()
                if 'text' in input_dict.keys():
                    input_dict['content'] = input_dict.pop('text')
                if role == 'model':
                    role = 'assistant'
                parsed_input = [{
                    "role": role,
                    "content": [{"type": "text", "text": input_dict['content']}],
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

    def _check_template(self):
        template_fields = set(re.findall(r'{([^{}]*)}', self.template))
        missing_fields = set(self.prompt_required_fields) - template_fields
        if len(missing_fields) > 0:
            raise ValueError(f'{self.name} fields {missing_fields} are required in prompt template.')

    async def generate(self, inputs: dict | list[dict]):
        if isinstance(inputs, dict):
            inputs = [inputs]
        response = await self.engine.generate(self._parse_inputs(inputs), self.generation_params)
        decoded_output = response.content[0].text
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
                decoded_output_batch.append(response.content[0].text)

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
                decoded_output_batch.append(response.content[0].text)
                self.time_manager['input_token'] += response.usage.input_tokens
                self.time_manager['output_token'] += response.usage.output_tokens
                self.time_manager['num_requests'] += 1

            yield decoded_output_batch

    async def generate_batched(self, inputs: list[list], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')

    async def generate_batched_streaming(self, inputs: list[list], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')


class AnthropicVisionWorker(AnthropicWorker):
    """
    A specialized worker class for handling vision-based tasks with Anthropic's API.
    Extends AnthropicWorker to include image processing capabilities.
    This class provides functionality to:
    - Process and resize images to meet Anthropic's API requirements
    - Handle multiple image input formats (URL, file path, PIL Image)
    - Convert images to base64 format for API transmission
    Parameters
    ----------
    engine : AnthropicClient
        The Anthropic API client instance
    prompt_template : str, optional
        Template string for formatting prompts
    prompt_default_fields : dict, optional
        Default field values for prompt template
    prompt_required_fields : list[str], optional
        List of required fields for prompt template
    system_prompt : str, optional
        System-level prompt for the model
    generation_params : dict, optional
        Parameters for text generation (defaults to anthropic_default_params)
    image_max_size : int, optional
        Maximum dimension (width or height) for images (default: 1568)
    Methods
    -------
    _resize_image(image: Image.Image)
        Resizes an image while maintaining aspect ratio if it exceeds max_size
    _process_image(image: str | Path | Image.Image)
        Processes different image input formats into base64 for API transmission
    _get_prompt(**kwargs)
        Generates a formatted prompt including image content for the API
    Notes
    -----
    - Images larger than image_max_size will be automatically resized
    - Supported image input formats: URL, file path, PIL Image
    - Images are converted to PNG format when processed from PIL Image objects
    """

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

    def _process_image(self, image: str | Path | Image.Image):
        base64_image, media_type = load_image_for_llm(image, self.image_max_size)
        return dict(
            type='image',
            source=dict(type='base64', media_type=media_type, data=base64_image))

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
