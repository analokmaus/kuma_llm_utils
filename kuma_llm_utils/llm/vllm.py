import re
import asyncio
import uuid
import torch

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

from ..utils import create_batches, safe_async_run
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
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024
)


class vLLMEngineAsync(AbstractLLMEngine):

    def __init__(
            self,
            vllm_params: dict = vllm_model_default_params):
        super().__init__()
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**vllm_params))

    async def _async_generate(self, prompt, sampling_params, request_id):
        tokenizer = await self.get_tokenizer()
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id)
        output = None
        async for request_output in results_generator:
            output = request_output
        prompt = output.prompt
        output = [output.text for output in output.outputs][0]
        input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
        self.logger.info(f'{self.name} input tokens = {input_tokens} | output tokens = {output_tokens}')
        return {
            'input': prompt,
            'output': output,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }

    async def get_tokenizer(self):
        tokenizer = await self.engine.get_tokenizer()
        return tokenizer

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
        if self.system_prompt is not None:
            messages.insert(0, {
                "role": "system",
                "content": self.system_prompt
            })

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt_text

    def _get_prompt_for_chat(self, chat_history: list[dict], question: str):
        question = self._fill_template(question=question)
        new_chat_history = chat_history.copy()
        new_chat_history.append({
            'role': 'user',
            'content': question
        })
        if self.system_prompt is not None:
            new_chat_history.insert(0, {
                "role": "system",
                "content": self.system_prompt
            })
        prompt_text = self.tokenizer.apply_chat_template(new_chat_history, tokenize=False, add_generation_prompt=True)
        return prompt_text

    def _check_template(self):
        template_fields = set(re.findall(r'{([^{}]*)}', self.template))
        missing_fields = set(self.prompt_required_fields) - template_fields
        if len(missing_fields) > 0:
            raise ValueError(f'{self.name} fields {missing_fields} are required in prompt template.')

    def _remove_reasoning_tag(self, text: str):
        text = re.sub(r'<think>[\s\S]*?<\/think>', '', text, flags=re.DOTALL)
        text = text.strip()
        return text

    async def generate(self, **kwargs):
        response = await self.engine.generate(self._get_prompt(**kwargs), self.generation_params)
        decoded_output = response['output']
        if self.remove_reasoning_tag:
            decoded_output = self._remove_reasoning_tag(decoded_output)
        return decoded_output

    async def generate_batched(self, inputs: list[dict], batch_size: int = 4):
        messages = [self._get_prompt(**kwargs) for kwargs in inputs]
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

    async def generate_batched_streaming(self, inputs: list[dict], batch_size: int = 4):
        messages = [self._get_prompt(**kwargs) for kwargs in inputs]
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

    async def chat(self, chat_history: str, question: str):
        response = await self.engine.generate(
            self._get_prompt(chat_history=chat_history, question=question), self.generation_params)
        decoded_output = response['output']
        if self.remove_reasoning_tag:
            decoded_output = self._remove_reasoning_tag(decoded_output)
        return decoded_output
