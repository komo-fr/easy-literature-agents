import re


def clean_json_string(s: str) -> str:
    # 1. ```json ... ``` を除去
    s = s.strip().lstrip("```json").rstrip("```").strip()

    # 2. JSONオブジェクト末尾のカンマを除去
    s = re.sub(r",(\s*})", r"\1", s)  # 末尾が }, の形を } にする
    s = re.sub(r",(\s*])", r"\1", s)  # 末尾が ], の形を ] にする
    return s
