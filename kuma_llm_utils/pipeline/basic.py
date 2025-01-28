import json
import re

from .abstract import AbstractPipeline
from ..llm.abstract import AbstractLLMWorker
from ..utils import load_json_string


class LLMModule(AbstractPipeline):

    def __init__(
            self,
            llm: AbstractLLMWorker,
            mode: str = 'text',
            output_key: str | None = None,
            input_keys: list = ['question'],
            output_keys: list = ['question']):
        super().__init__()
        self.llm = llm
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.mode = mode
        self.output_key = output_key
        if self.mode == 'text':
            if self.output_key is None:
                raise ValueError(f'{self.name} output_key must be set.')
        elif self.mode == 'json':
            pass
        else:
            raise ValueError(f'{self.name} unknown mode: {self.mode}.')

    async def generate(self, inputs: dict) -> dict:
        self.check_format(inputs=inputs)
        outputs = inputs.copy()
        response = await self.llm.generate([inputs])
        if self.mode == 'text':
            outputs[self.output_key] = response
        elif self.mode == 'json':
            outputs.update(load_json_string(response))
        self.check_format(outputs=outputs)
        return outputs


class JsonToText(AbstractPipeline):
    def __init__(
            self,
            json_key: str,
            template: str,
            output_key: str | None = None,
            verbose: bool = False,
            input_keys: list = ['question'],
            output_keys: list = ['question']):
        super().__init__()
        self.json_key = json_key
        self.template = template
        if output_key is None:
            self.output_key = json_key
        else:
            self.output_key = output_key
        self.verbose = verbose
        self.input_keys = input_keys
        self.output_keys = output_keys

    def _apply_template(self, json_dict: dict):
        if not hasattr(self, '_template_fields'):
            self._template_fields = re.findall(r'{([^{}]*)}', self.template)
        return self.template.format(**{k: json_dict[k] for k in self._template_fields})

    async def generate(self, inputs: dict):
        self.check_format(inputs=inputs)
        outputs = inputs.copy()
        output_text = ''
        for json_dict in inputs[self.json_key]:
            output_text += self._apply_template(json_dict)
        if self.verbose:
            self.logger.info(output_text)
        outputs[self.output_key] = output_text
        self.check_format(outputs=outputs)
        return outputs


class UpdateKey(AbstractPipeline):
    def __init__(
            self,
            update_dict: dict,
            input_keys: list = ['question'],
            output_keys: list = ['question']):
        super().__init__()
        self.update_dict = update_dict
        self.input_keys = input_keys
        self.output_keys = output_keys

    async def generate(self, inputs: dict):
        self.check_format(inputs=inputs)
        outputs = inputs.copy()
        outputs.update(self.update_dict)
        self.check_format(outputs=outputs)
        return outputs


class Compose(AbstractPipeline):
    def __init__(
            self,
            pipelines: list[AbstractPipeline],
            input_keys: list = ['question'],
            output_keys: list = ['question']):
        super().__init__()
        self.pipelines = pipelines
        self.input_keys = input_keys
        self.output_keys = output_keys

    async def generate(self, inputs: dict):
        self.check_format(inputs=inputs)
        for p in self.pipelines:
            inputs = await p.generate(inputs)
        return inputs
