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

    async def generate(self, **kwargs):
        pass

    async def generate_batched(self, inputs, batch_size):
        pass

    async def generate_batched_streaming(self, inputs, batch_size):
        pass

    async def generate_parallel(self, inputs, batch_size):
        pass

    async def generate_parallel_streaming(self, inputs, batch_size):
        pass

    async def chat(self, chat_history, question):
        pass
