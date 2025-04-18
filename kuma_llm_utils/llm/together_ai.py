from together import Together
import os
import re
import asyncio
from pathlib import Path
from PIL import Image
from collections import defaultdict

from ..utils import create_batches
from .abstract import AbstractLLMEngine, AbstractLLMWorker
from .utils import LimitManager
from .openai import OpenAIWorker


together_ai_default_params = dict(
    model='deepseek-ai/DeepSeek-R1',
    max_tokens=8192,
    temperature=0.5,
    top_p=0.1
)

together_ai_default_limits = {
    'deepseek-ai/DeepSeek-R1': [{'request': 220, 'token': 240000, 'reset_cycle': 60}],
    'deepseek-ai/DeepSeek-V3': [{'request': 1500, 'token': 240000, 'reset_cycle': 60}],
}


class TogetherAIClient(AbstractLLMEngine):
    
    def __init__(
            self,
            api_key: str = None,
            usage_limits: dict = together_ai_default_limits):
        super().__init__()
        self.api_key = api_key
        self.__api_key = self._retrieve_api_key()
        self.client = Together(api_key=self.__api_key)
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
        api_key = os.environ.get("TOGETHER_AI_API_KEY")
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
        responses = await asyncio.gather(*tasks)
        return responses


class TogetherAIWorker(OpenAIWorker):

    def __init__(
            self,
            engine: TogetherAIClient,
            prompt_template: str = '',
            prompt_default_fields: dict = {},
            prompt_required_fields: list[str] = [],
            system_prompt: str = None,
            generation_params: dict = together_ai_default_params
    ):
        super().__init__(
            engine=engine,
            prompt_template=prompt_template,
            prompt_default_fields=prompt_default_fields,
            prompt_required_fields=prompt_required_fields,
            system_prompt=system_prompt,
            generation_params=generation_params
        )

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
                    "content": input_dict['content'],
                }]
            else:
                raise ValueError(f'{self.name} role {role} is not supported.')
            parsed_inputs.extend(parsed_input)
        if self.system_prompt is not None:
            parsed_inputs.insert(0, {
                "role": "system",
                "content": self.system_prompt
            })
        return parsed_inputs
