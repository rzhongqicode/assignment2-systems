import json

import pandas as pd


def generate_table_from_dict(input: list[dict], mode: str) -> str:
    df = pd.DataFrame(input)
    if mode == "latex":
        output = df.to_latex(index=False)  # remove meaningless index
    elif mode == "markdown":
        output = df.to_markdown(index=False)
    else:
        print("Mode Error! Please choose form 'latex' or 'markdown'")
        output = ""
    return output


def generate_table_from_json(file_path: str, mode: str) -> str:
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # 假设你的 JSON 文件内容就是一个字典列表 (list[dict])
        return generate_table_from_dict(data, mode)

    except FileNotFoundError:
        print(f"Error: 找不到文件 {file_path}")
        return ""
    except json.JSONDecodeError:
        print(f"Error: {file_path} 不是一个有效的 JSON 文件")
        return ""
