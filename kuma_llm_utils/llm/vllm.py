import re
import asyncio
import uuid
import torch
from pathlib import Path
from PIL import Image

from transformers import AutoProcessor
from vllm.inputs import TextPrompt
from vllm.multimodal import MultiModalDataBuiltins
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

from ..utils import create_batches, safe_async_run, load_image_for_llm
from .abstract import AbstractLLMEngine, AbstractLLMWorker


vllm_model_default_params = dict(
    model='google/gemma-2-2b-it',
    dtype=torch.bfloat16,
    gpu_memory_utilization=0.8,
    quantization='bitsandbytes',
    load_format='bitsandbytes',
    trust_remote_code=True,
    disable_log_requests=True
)


vllm_sampling_default_params = dict(
    temperature=0.,
    top_p=1.,
    max_tokens=1024
)


class vLLMEngineAsync(AbstractLLMEngine):
    """vLLM (very Large Language Model) Engine for asynchronous text generation.
    This class implements an asynchronous interface for the vLLM engine, providing methods
    for both single and batched text generation using large language models.
    Attributes:
        engine (AsyncLLMEngine): The underlying vLLM engine instance for text generation.
        logger (Logger): Logger instance for tracking generation statistics.
        name (str): Name identifier for the engine.
    Methods:
        generate(prompt: str, sampling_params: SamplingParams) -> dict:
            Generates text from a single prompt asynchronously.
        generate_batched(prompts: list[str], sampling_params: SamplingParams) -> list[dict]:
            Generates text from multiple prompts in parallel.
        get_tokenizer() -> PreTrainedTokenizer:
            Returns the tokenizer associated with the engine.
    Returns:
        For both generate and generate_batched methods, returns a dictionary or list of dictionaries with:
            - input: Original prompt text
            - output: Generated text
            - input_tokens: Number of tokens in the input
            - output_tokens: Number of tokens in the output
    """

    def __init__(
            self,
            vllm_params: dict = vllm_model_default_params):
        super().__init__()
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**vllm_params))
        self._model_name = vllm_params['model']

    async def _async_generate(self, prompt, sampling_params, request_id):
        tokenizer = await self.get_tokenizer()
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id)
        response = None
        async for request_output in results_generator:
            response = request_output
        prompt = response.prompt
        output = response.outputs[0].text
        input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
        self.logger.info(f'{self.name} input tokens = {input_tokens} | output tokens = {output_tokens}')
        response_dict = {
            'input': prompt,
            'output': output,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        }
        if hasattr(response.outputs[0], 'logprobs'):
            response_dict['logprobs'] = response.outputs[0].logprobs
        return response_dict

    async def get_tokenizer(self):
        tokenizer = await self.engine.get_tokenizer()
        return tokenizer
    
    def get_preprocessor(self):
        preprocessor = AutoProcessor.from_pretrained(self._model_name)
        return preprocessor

    async def generate(self, prompt: str, sampling_params: SamplingParams):
        response = await asyncio.create_task(self._async_generate(
            prompt=prompt, sampling_params=sampling_params, request_id=uuid.uuid4()))
        return response

    async def generate_batched(self, prompts: list[str], sampling_params: SamplingParams):
        tasks = [asyncio.create_task(
            self._async_generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=uuid.uuid4())) for prompt in prompts]
        responses = [await task for task in tasks]
        return responses


class vLLMWorkerAsync(AbstractLLMWorker):
    """A worker class for asynchronous text generation using vLLM.
    This class implements an asynchronous interface for text generation using vLLM engine.
    It handles prompt templating, batched generation, and streaming outputs.
    Args:
        engine (vLLMEngineAsync): The async vLLM engine instance for text generation.
        prompt_template (str, optional): Template string for formatting prompts. Defaults to ''.
        prompt_default_fields (dict, optional): Default field values for the prompt template. Defaults to {}.
        prompt_required_fields (list[str], optional): Required fields that must exist in template. Defaults to [].
        system_prompt (str, optional): System prompt to prepend to all generations. Defaults to None.
        generation_params (dict, optional): Parameters for text generation. Defaults to vllm_sampling_default_params.
        remove_reasoning_tag (bool, optional): Whether to remove <think> tags from output. Defaults to True.
    Attributes:
        engine: The vLLM engine instance
        template: The prompt template string
        prompt_default_fields: Default values for prompt template fields
        prompt_required_fields: Required fields for prompt template
        system_prompt: System prompt for all generations
        generation_params: Generation parameters as SamplingParams
        remove_reasoning_tag: Flag to remove reasoning tags
        tokenizer: The tokenizer from the engine
        is_async: Always True for this class
    Methods:
        generate(inputs): Generate text for a single input
        generate_batched(inputs, batch_size): Generate text for multiple inputs in batches
        generate_batched_streaming(inputs, batch_size): Stream generated text for batched inputs
        generate_parallel: Alias for generate_batched
        generate_parallel_streaming: Alias for generate_batched_streaming
    """
    
    def __init__(
            self,
            engine: vLLMEngineAsync,
            prompt_template: str = '',
            prompt_default_fields: dict = {},
            prompt_required_fields: list[str] = [],
            system_prompt: str = None,
            generation_params: dict = vllm_sampling_default_params,
            remove_reasoning_tag: bool = True):
        super().__init__()
        self.engine = engine
        self.template = prompt_template
        self.prompt_default_fields = prompt_default_fields
        self.prompt_required_fields = prompt_required_fields
        self.system_prompt = system_prompt
        self.generation_params = SamplingParams(**generation_params)
        self.remove_reasoning_tag = remove_reasoning_tag
        self.tokenizer = safe_async_run(self.engine.get_tokenizer())
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
                "content": self._fill_template(**kwargs)
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
                parsed_input = [{"role": "assistant", "content": input_dict['text']}]
            else:
                raise ValueError(f'{self.name} role {role} is not supported.')
            parsed_inputs.extend(parsed_input)
        if self.system_prompt is not None:
            parsed_inputs.insert(0, {"role": "system", "content": self.system_prompt})
        prompt_text = self.tokenizer.apply_chat_template(
            parsed_inputs, tokenize=False, add_generation_prompt=True)
        return prompt_text
    
    def _parse_logprobs(self, logprobs: list):
        logprobs_json = []
        for logprob in logprobs:
            if logprob is None:
                continue
            for v in logprob.values():
                logprobs_json.append(dict(
                    token=v.decoded_token,
                    logprob=v.logprob
                ))
        return logprobs_json

    def _check_template(self):
        template_fields = set(re.findall(r'{([^{}]*)}', self.template))
        missing_fields = set(self.prompt_required_fields) - template_fields
        if len(missing_fields) > 0:
            raise ValueError(f'{self.name} fields {missing_fields} are required in prompt template.')

    def _remove_reasoning_tag(self, text: str):
        text = re.sub(r'<think>[\s\S]*?<\/think>', '', text, flags=re.DOTALL)
        text = text.strip()
        return text

    async def generate(self, inputs: dict | list[dict], return_logprobs: bool = False):
        if isinstance(inputs, dict):
            inputs = [inputs]
        response = await self.engine.generate(self._parse_inputs(inputs), self.generation_params)
        decoded_output = response['output']
        if self.remove_reasoning_tag:
            decoded_output = self._remove_reasoning_tag(decoded_output)
        if return_logprobs:
            return decoded_output, self._parse_logprobs(response['logprobs'])
        else:
            return decoded_output

    async def generate_batched(self, inputs: list[list], batch_size: int = 4):
        messages = [self._parse_inputs(inputs_item) for inputs_item in inputs]
        messages_batches = create_batches(messages, batch_size)
        decoded_outputs = []
        for batch_i, batch in enumerate(messages_batches):
            self.logger.info(f'{self.name} processing batch {batch_i+1}/{len(messages_batches)}')
            responses = await self.engine.generate_batched(batch, self.generation_params)
            if self.remove_reasoning_tag:
                decoded_output_batch = [self._remove_reasoning_tag(res['output']) for res in responses]
            else:
                decoded_output_batch = [res['output'] for res in responses]
            decoded_outputs.extend(decoded_output_batch)

        return decoded_outputs

    async def generate_batched_streaming(self, inputs: list[list], batch_size: int = 4):
        messages = [self._parse_inputs(inputs_item) for inputs_item in inputs]
        messages_batches = create_batches(messages, batch_size)
        for batch_i, batch in enumerate(messages_batches):
            self.logger.info(f'{self.name} processing batch {batch_i+1}/{len(messages_batches)}')
            responses = await self.engine.generate_batched(batch, self.generation_params)
            if self.remove_reasoning_tag:
                decoded_output_batch = [self._remove_reasoning_tag(res['output']) for res in responses]
            else:
                decoded_output_batch = [res['output'] for res in responses]
            yield decoded_output_batch

    generate_parallel = generate_batched
    generate_parallel_streaming = generate_batched_streaming


class vLLMVisionWorkerAsync(vLLMWorkerAsync):

    def __init__(
            self,
            engine: vLLMEngineAsync,
            prompt_template: str = '',
            prompt_default_fields: dict = {},
            prompt_required_fields: list[str] = [],
            system_prompt: str = None,
            generation_params: dict = vllm_sampling_default_params,
            remove_reasoning_tag: bool = True,
            image_max_size: int = 1568):
        super().__init__(
            engine=engine,
            prompt_template=prompt_template,
            prompt_default_fields=prompt_default_fields,
            prompt_required_fields=prompt_required_fields,
            system_prompt=system_prompt,
            generation_params=generation_params,
            remove_reasoning_tag=remove_reasoning_tag)
        self.image_max_size = image_max_size
        self.preprocessor = self.engine.get_preprocessor()

    def _process_image(self, image: str | Path | Image.Image):
        pil_image = load_image_for_llm(image, self.image_max_size, return_pil=True)
        return dict(type='image', image=pil_image)

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
    
    def _parse_inputs(self, inputs: list[dict]):
        parsed_inputs = []
        for input_dict in inputs:
            if 'role' not in input_dict.keys():
                input_dict['role'] = 'user'
            role = input_dict.pop('role')
            if role == 'user':
                parsed_input = self._get_prompt(**input_dict)
            elif role in ['assistant', 'model']:
                parsed_input = [{"role": "assistant", "content": input_dict['text']}]
            else:
                raise ValueError(f'{self.name} role {role} is not supported.')
            parsed_inputs.extend(parsed_input)
        if self.system_prompt is not None:
            parsed_inputs.insert(0, {"role": "system", "content": self.system_prompt})
        prompt_text = self.preprocessor.apply_chat_template(
            parsed_inputs, tokenize=False, add_generation_prompt=True)
        images = []
        for msg in parsed_inputs:
            if msg["role"] == "user":
                for chunk in msg["content"]:
                    if chunk["type"] == "image":
                        images.append(chunk["image"])
        mm_data = MultiModalDataBuiltins(image=images)
        prompt = TextPrompt(prompt=prompt_text, multi_modal_data=mm_data)
        return prompt
