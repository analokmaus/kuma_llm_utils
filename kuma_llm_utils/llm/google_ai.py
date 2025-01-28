from google import genai
from google.genai import types
import os
import re
import asyncio
from pathlib import Path
from PIL import Image
from collections import defaultdict

from .abstract import AbstractLLMEngine, AbstractLLMWorker
from .utils import LimitManager
from ..utils import create_batches, load_image_for_llm


google_ai_default_limits = {
    "gemini-2.0-flash": [
        {"request": 15, "reset_cycle": 60},
        {"request": 1500, "reset_cycle": 3600 * 24},
    ]
}
google_ai_default_params = {
    'model': 'gemini-2.0-flash',
    'max_out_tokens': 1024,
    'temperature': 0.5,
    'top_p': 0.5,
}


class GoogleAIClient(AbstractLLMEngine):
    """A client class for interacting with Google AI's generative models.
    This class implements the AbstractLLMEngine interface for Google AI services,
    providing methods to generate content using Google's AI models with built-in
    usage limits and rate limiting functionality.
    Attributes:
        api_key (str): The API key for authentication with Google AI services.
        client (genai.Client): The Google AI client instance.
        usage_limits (defaultdict): A dictionary of model-specific usage limits and their managers.
    Args:
        api_key (str, optional): API key for Google AI services. If not provided,
            will attempt to retrieve from environment variables.
        usage_limits (dict, optional): Dictionary defining usage limits for different models.
            Defaults to google_ai_default_limits.
    Methods:
        generate(message, generation_params): Generate content for a single message.
        generate_parallel(messages, generation_params): Generate content for multiple messages in parallel.
    Example:
        >>> client = GoogleAIClient(api_key="your-api-key")
        >>> response = await client.generate("Hello!", {"model": "gemini-pro"})
    """
    
    def __init__(
            self,
            api_key: str = None,
            usage_limits: dict = google_ai_default_limits):
        super().__init__()
        self.api_key = api_key
        self.__api_key = self._retrieve_api_key()
        self.client = genai.Client(api_key=self.__api_key)
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
        api_key = os.environ.get("GOOGLE_AI_API_KEY")
        if self.api_key is not None:
            api_key = self.api_key
            self.logger.warning(f'{self.name} the API key in the environment variable has been overridden by the argument.')
        if api_key is None:
            self.logger.critical(f'{self.name} API key not found.')
        return api_key

    async def _async_generate(self, message, generation_params):
        await self._call_time_manager()
        generation_params2 = generation_params.copy()
        model_name = generation_params2.pop('model')
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model_name,
            contents=message,
            config=types.GenerateContentConfig(**generation_params2)
        )
        self._update_counter(
            generation_params['model'],
            {'request': 1, 'input_token': response.usage_metadata.prompt_token_count,
             'output_token': response.usage_metadata.candidates_token_count})
        return response

    async def generate(self, message, generation_params):
        return await asyncio.create_task(self._async_generate(message, generation_params))

    async def generate_parallel(self, messages, generation_params):
        tasks = [asyncio.create_task(
            self._async_generate(message, generation_params)) for message in messages]
        responses = [await task for task in tasks]
        return responses
    

class GoogleAIWorker(AbstractLLMWorker):
    """A worker class for interacting with Google AI language models.
    This class implements the AbstractLLMWorker interface for Google AI models,
    providing methods for text generation with various input configurations.
    Args:
        engine (GoogleAIClient): The Google AI client instance for making API calls.
        prompt_template (str, optional): Template string for formatting prompts. Defaults to ''.
        prompt_default_fields (dict, optional): Default fields to use in prompt template. Defaults to {}.
        prompt_required_fields (list[str], optional): List of required fields in prompt template. Defaults to [].
        system_prompt (str, optional): System prompt to prepend to all interactions. Currently not supported. Defaults to None.
        generation_params (dict, optional): Parameters for text generation. Defaults to google_ai_default_params.
        remove_reasoning_tag (bool, optional): Whether to remove reasoning tags from output. Defaults to True.
    Attributes:
        is_async (bool): Indicates that this worker supports async operations.
    Methods:
        generate(inputs: list[dict]) -> str:
            Generate text from a list of input dictionaries asynchronously.
        generate_parallel(inputs: list[dict], batch_size: int = 4) -> list[str]:
            Generate text for multiple inputs in parallel with batching.
        generate_parallel_streaming(inputs: list[dict], batch_size: int = 4) -> Generator[list[str]]:
            Stream generated text for multiple inputs in parallel with batching.
    Raises:
        ValueError: If required fields are missing from the prompt template.
        NotImplementedError: If system prompt is provided (not supported) or when using unimplemented batch methods.
    Note:
        - The class supports both single and parallel text generation.
        - Batch processing is done through the generate_parallel methods.
        - System prompts are currently not supported.
        - Input dictionaries should contain either 'role': 'user' with template fields or 'role': 'assistant' with 'text'.
    """

    def __init__(
            self,
            engine: GoogleAIClient,
            prompt_template: str = '',
            prompt_default_fields: dict = {},
            prompt_required_fields: list[str] = [],
            system_prompt: str = None,
            generation_params: dict = google_ai_default_params,
            remove_reasoning_tag: bool = True):
        super().__init__()
        self.engine = engine
        self.template = prompt_template
        self.prompt_default_fields = prompt_default_fields
        self.prompt_required_fields = prompt_required_fields
        self.system_prompt = system_prompt
        self.generation_params = generation_params
        self.remove_reasoning_tag = remove_reasoning_tag
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
            types.Content(
                role='user',
                parts=[types.Part.from_text(self._fill_template(**kwargs))]
            )
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
            elif role == 'assistant':
                parsed_input = [
                    types.Content(
                        role='assistant',
                        parts=[types.Part.from_text(input_dict['text'])]
                    )
                ]
            else:
                raise ValueError(f'{self.name} role {role} is not supported.')
            parsed_inputs.extend(parsed_input)
        if self.system_prompt is not None:
            raise NotImplementedError(f'{self.name} system prompt is not supported.')
        return parsed_inputs

    def _check_template(self):
        template_fields = set(re.findall(r'{([^{}]*)}', self.template))
        missing_fields = set(self.prompt_required_fields) - template_fields
        if len(missing_fields) > 0:
            raise ValueError(f'{self.name} fields {missing_fields} are required in prompt template.')

    async def generate(self, inputs: list[dict]):
        response = await self.engine.generate(self._parse_inputs(inputs), self.generation_params)
        decoded_output = response.text
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
                decoded_output_batch.append(response.text)

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
                decoded_output_batch.append(response.text)

            yield decoded_output_batch

    async def generate_batched(self, inputs: list[dict], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')

    async def generate_batched_streaming(self, inputs: list[dict], batch_size: int = 4):
        raise NotImplementedError(f'{self.name} batch inference is not yet implemented.')


class GoogleAIVisionWorker(GoogleAIWorker):
    """A worker class for handling vision-based tasks using Google's AI API.
    This class extends GoogleAIWorker to provide functionality for processing and analyzing images
    using Google's AI services. It handles image resizing, processing of different image input formats,
    and preparation of prompts with image content.
    Parameters
    ----------
    engine : GoogleAIClient
        The Google AI client instance for making API calls
    prompt_template : str, optional
        Template string for formatting prompts
    prompt_default_fields : dict, optional
        Default fields to use in prompt template
    prompt_required_fields : list[str], optional
        List of required fields for prompt template
    system_prompt : str, optional
        System-level prompt to use
    generation_params : dict, optional
        Parameters for generation using Google AI API
    image_max_size : int, optional
        Maximum size (in pixels) for image dimension, default is 2048
    Attributes
    ----------
    image_max_size : int
        Maximum allowed size for image dimensions
    Methods
    -------
    _resize_image(image: Image.Image)
        Resizes an image while maintaining aspect ratio if it exceeds maximum size
    _process_image(image: str | Path | Image.Image)
        Processes different types of image inputs into base64 format
    _get_prompt(**kwargs)
        Prepares the prompt with image content for API submission
    Notes
    -----
    - Supports image input as URL, file path, or PIL Image object
    - Automatically resizes images that exceed the maximum dimension while preserving aspect ratio
    - Converts images to base64 format for API submission
    """

    def __init__(
            self,
            engine: GoogleAIClient,
            prompt_template: str = '',
            prompt_default_fields: dict = {},
            prompt_required_fields: list[str] = [],
            system_prompt: str = None,
            generation_params: dict = google_ai_default_params,
            image_max_size: int = 2048
    ):
        super().__init__(
            engine,
            prompt_template,
            prompt_default_fields,
            prompt_required_fields,
            system_prompt,
            generation_params,
        )
        self.image_max_size = image_max_size

    def _process_image(self, image: str | Path | Image.Image):
        base64_image, media_type = load_image_for_llm(image, self.image_max_size)
        return types.Part.from_bytes(base64_image, mime_type=media_type)

    def _get_prompt(self, **kwargs):
        image_contents = []
        image_keys = [k for k in kwargs.keys() if re.match(r'^image(\d)*$', k)]
        for k in image_keys:
            image = kwargs.pop(k)
            image_contents.append(self._process_image(image))
        messages = [
            types.Content(
                role='user',
                parts=[
                    types.Part.from_text(self._fill_template(**kwargs))
                ] + image_contents
            )
        ]
        return messages
