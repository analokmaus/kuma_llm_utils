from ..logger import DefaultLogger


class AbstractPipeline:
    def __init__(self):
        self.logger = DefaultLogger().get_logger()
        self.name = f'{self.__class__.__name__} | '

    async def generate(self, inputs: dict) -> dict:
        pass

    def check_format(self, inputs: dict = None, outputs: dict = None):
        if inputs is not None:
            for k in self.input_keys:
                assert k in inputs.keys()
        if outputs is not None:
            for k in self.output_keys:
                assert k in outputs.keys()
