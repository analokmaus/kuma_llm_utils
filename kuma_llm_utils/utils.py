import asyncio
import base64
from io import BytesIO
from PIL import Image
import urllib.parse
from concurrent.futures import ThreadPoolExecutor


'''
Misc.
'''
def create_batches(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def identity(x):
    return x


'''
Async utils
'''
def run_async_in_thread(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine)
    loop.close()
    return result


def safe_async_run(task):
    with ThreadPoolExecutor(1) as pool:
        responses = pool.submit(run_async_in_thread, task).result()
    return responses


async def async_zip(sync_gen, async_gen):
    sync_iter = iter(sync_gen)
    async for async_item in async_gen:
        try:
            sync_item = next(sync_iter)
            yield (sync_item, async_item)
        except StopIteration:
            break


'''
Image processing
'''
def is_url(path: str):
    url_schemes = ('http', 'https', 'ftp', 'ftps', 'file', 'mailto', 'data')
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme in url_schemes:
        return True
    return False


def pil_to_base64(img: Image.Image, img_format: str = "png"):
    buffer = BytesIO()
    img.save(buffer, img_format)
    img_str = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return img_str
