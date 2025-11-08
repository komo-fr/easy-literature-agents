import json
from datetime import datetime
from pathlib import Path


def read_input_text(file_path: str) -> str:
    """入力テキストを読み込む関数"""
    ENCODINGS = ["utf-8", "shift_jis", "cp932", "euc_jp"]
    for encoding in ENCODINGS:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        f"ファイルを読み込めませんでした。対応しているエンコーディング: {', '.join(ENCODINGS)}"
    )


def save_output_text(
    state
) -> None:
    """出力テキストを保存する関数"""
    OUTPUT_DIR = Path(__file__).parent / "output"
    LOG_DIR = OUTPUT_DIR / "logs"

    # 出力データの作成
    output_data = {
        "draft_1": state.rewritten_sentence.rewritten_text,
        "draft_2": state.suggestion_applier_output.model_dump(),
        "draft_3": state.styled_text,
        "child_feedback": state.child_feedback.model_dump(),
        "adult_feedback": state.adult_feedback.model_dump(),
        "suggester_output": state.suggester_output.model_dump(),
        "highlight_output": state.highlight_output.model_dump(),
        "highlight_with_web_output": state.highlight_with_web_output.model_dump(),
    }

    # メインの出力ファイルに保存（JSON形式）
    output_path = Path(OUTPUT_DIR) / "output_all.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # テキストファイルとしても保存
    with open(output_path, "w", encoding="utf-8") as f:
        # 本文を書き込み
        f.write(str(state))

    # 各稿を個別のファイルとして保存
    texts = [state.rewritten_sentence.rewritten_text,
             state.suggestion_applier_output.revised_text,
             state.styled_text]

    for i, content in enumerate(texts):
        filename = f"draft_{i + 1}.txt"
        with open(Path(OUTPUT_DIR) / filename, "w", encoding="utf-8") as f:
            f.write(content)

    # ログディレクトリにタイムスタンプ付きで保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(LOG_DIR) / f"rewritten_text_{timestamp}"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
