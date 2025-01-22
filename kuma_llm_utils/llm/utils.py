import asyncio
import time
import logging


class LimitManager:

    def __init__(
            self,
            limits: dict,
            logger: logging.Logger
    ):
        self.limits = limits
        self.logger = logger
        self.state = {}
        for k in self.limits.keys():
            if k == 'reset_cycle':
                continue
            self.state[k] = 0
        self.state['reference_time'] = time.time()

    def add(self, add: dict):
        for k, v in add.items():
            if k in self.state.keys():
                self.state[k] = self.state[k] + v

    async def check(self):
        current_time = time.time()
        elapsed_time = current_time - self.state['reference_time']
        if elapsed_time >= self.limits['reset_cycle']:
            for k in self.state.keys():
                self.state[k] = 0
            self.state['reference_time'] = current_time
        for k in self.limits.keys():
            if k == 'reset_cycle' or self.limits[k] is None:
                continue
            if self.state[k] >= self.limits[k]:
                wait_time = self.limits['reset_cycle'] - elapsed_time + 0.1
                self.logger.info(
                    f'{k} ({self.state[k]}) reaches the limit. waiting for {wait_time:.1f} secs.')
                await asyncio.sleep(wait_time)
