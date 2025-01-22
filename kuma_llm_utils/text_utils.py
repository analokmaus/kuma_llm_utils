import mojimoji
import json
import re


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
