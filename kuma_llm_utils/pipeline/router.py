import re

from .abstract import AbstractPipeline
from ..llm.abstract import AbstractLLMWorker


class LLMRouter(AbstractPipeline):

    def __init__(
            self,
            route_llm: AbstractLLMWorker,
            job_pipeline: dict[str, AbstractPipeline],
            input_keys: list = ['question'],
            output_keys: list = ['question']):
        super().__init__()
        self.route_llm = route_llm
        self.job_pipeline = job_pipeline
        self.input_keys = input_keys
        self.output_keys = output_keys
        if 'default' not in job_pipeline.keys():
            raise ValueError(f'{self.name} job_pipeline must have key "default".')

    async def generate(self, inputs: dict) -> dict:
        self.check_format(inputs=inputs)
        route_response = await self.route_llm.generate(**inputs)
        route = self._postprocess_route(route_response)
        self.logger.info(f'{self.name} route = {route}')
        if route not in self.job_pipeline.keys():
            self.logger.warning(f'{self.name} route "{route}" not found in job_dict.')
            route = 'default'
        outputs = await self.job_pipeline[route].generate(inputs)
        self.check_format(outputs=outputs)
        return outputs

    def _postprocess_route(self, route: str):
        return re.sub(r'\n', '', route).strip()


class KeyRouter(AbstractPipeline):

    def __init__(
            self,
            target_key: str,
            job_pipeline: dict[str, AbstractPipeline],
            input_keys: list = ['question'],
            output_keys: list = ['question']):
        super().__init__()
        self.target_key = target_key
        self.job_pipeline = job_pipeline
        self.input_keys = input_keys
        self.output_keys = output_keys
        for key in ['default', self.target_key]:
            if key not in job_pipeline.keys():
                raise ValueError(f'{self.name} job_pipeline must have key "{key}".')

    async def generate(self, inputs: dict) -> dict:
        self.check_format(inputs=inputs)
        if self.target_key in inputs.keys():
            if isinstance(inputs[self.target_key], list) and len(inputs[self.target_key]) == 0:
                route = 'default'
                self.logger.info(f'{self.name} key {self.target_key} is an empty list.')
            else:
                route = self.target_key
        else:
            route = 'default'
        self.logger.info(f'{self.name} route = {route}')
        outputs = await self.job_pipeline[route].generate(inputs)
        self.check_format(outputs=outputs)
        return outputs
