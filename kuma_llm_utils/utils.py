import asyncio
import base64
from io import BytesIO
from PIL import Image
import httpx
from pathlib import Path
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import mojimoji
import json
import re


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


def resize_image(image: Image.Image, max_size: int = 1536):
    width, height = image.size
    if max(width, height) < max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)
    image_resized = image.resize((new_width, new_height), Image.BICUBIC)
    return image_resized


def load_image_for_llm(image: Image.Image | str | Path, max_size: int = 1536) -> tuple[str, str]:
    if isinstance(image, str | Path):
        if is_url(image):
            pil_img = Image.open(BytesIO(httpx.get(image).content))
        else:
            pil_img = Image.open(image)
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise ValueError('Unsupported image type')
    pil_img = resize_image(pil_img, max_size)
    base64_image = pil_to_base64(pil_img, 'png')
    media_type = 'image/png'
    return base64_image, media_type


'''
Text processing
'''
def preprocess_text_jp(text: str):
    text = mojimoji.zen_to_han(text, kana=False)
    text = text.replace('\u3000', '')
    text = text.replace('\\n', '\n')
    return text


def load_json_string(text: str):
    pattern = re.compile(
        r'```(?:json)?\s*\n'  # ```json または ``` の開始
        r'(.*?)'              # コード内容を非貪欲でキャプチャ
        r'\n```',             # ``` で終了
        re.DOTALL             # 改行を含む任意の文字にマッチ
    )
    matches = pattern.findall(text)
    if len(matches) > 0:
        json_dict = json.loads(matches[0])
    else:
        json_dict = json.loads(text)
    return json_dict
