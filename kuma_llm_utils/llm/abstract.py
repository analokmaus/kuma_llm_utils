from ..logger import DefaultLogger


class AbstractLLMEngine:
    def __init__(self):
        self.logger = DefaultLogger().get_logger()
        self.name = f'{self.__class__.__name__} | '

    async def generate(self):
        pass

    async def generate_batched(self):
        pass

    async def get_tokenizer(self):
        pass


class AbstractLLMWorker:
    def __init__(self):
        self.logger = DefaultLogger().get_logger()
        self.name = f'{self.__class__.__name__} | '

    def _check_template(self):
        pass

    async def generate(self, inputs: list[dict]):
        pass

    async def generate_batched(self, inputs: list[list], batch_size):
        pass

    async def generate_batched_streaming(self, inputs: list[list], batch_size):
        pass

    async def generate_parallel(self, inputs: list[list], batch_size):
        pass

    async def generate_parallel_streaming(self, inputs: list[list], batch_size):
        pass
