import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from questionary import select

from file_io import read_input_text, save_output_text
from workflow import Workflow, WorkflowState


def get_text_from_meta_info(text: str) -> str:
    """メタ情報を切り取ってテキストを返す"""
    start_index = text.find("-"*55)
    end_index = text.find("-"*55, start_index + 1)
    return text[end_index+55:].lstrip("\n").lstrip("\u3000")

def choose_model():
    choices = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4.1", "gpt-4o", "gpt-5"]
    return select("使用するモデルを選んでください：", choices=choices).ask()


def create_feedback(run_id: str):
    # ユーザからのフィードバックを取得
    vote = select(
        "フィードバックを入力してください: ",
        choices=["good", "so-so", "bad", "skip"],
    ).ask()
    if vote == "skip":
        print("👋 フィードバックをスキップします")
        return
    comment = input("コメントを入力してください: ")

    client = Client()
    client.create_feedback(run_id, key="quality_vote", value=vote, comment=comment)
    print("📤 フィードバックを作成しました")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="リライト対象のテキストファイルを指定します。")
    parser.add_argument("input_file", help="リライト対象の入力ファイル（例: input/sangetsuki.txt）")
    parser.add_argument("--model", help="モデル名（省略時は選択式）")
    parser.add_argument("--trim", help="テキストの長さを調整する（例: --trim 100）")
    parser.add_argument("--style", help="文体の模倣元となるテキスト（省略時は模倣をスキップ）")
    args = parser.parse_args()
    model_name = args.model or choose_model()

    load_dotenv()
    print("🔍 リライトを開始します...")

    # 入力テキストの読み込み
    input_text = read_input_text(args.input_file)

    if args.style:
        print(f"文体の模倣元となるテキストを読み込みます: {args.style}")
        style_source_text = get_text_from_meta_info(read_input_text(args.style))
    else:
        print("文体の模倣をスキップします")
        style_source_text = None

    if args.trim:
        print(f"テキストの長さを調整します: {args.trim}")
        input_text = input_text[:int(args.trim)]
        if style_source_text:
            style_source_text = style_source_text[:int(args.trim)]

    # ワークフローの作成と実行
    workflow = Workflow(model_name=model_name)
    workflow.output_graph()

    with collect_runs() as run_cb:
        # ワークフローの実行
        final_state = workflow.run(input_text, style_source_text,
                                   langsmith_extra={"metadata": {"file_name": Path(args.input_file).name}})
        run_id = run_cb.traced_runs[0].id

    # 結果の保存
    final_state = WorkflowState.model_validate(final_state)
    save_output_text(final_state)
    print("✨ リライトが完了しました！")
    # ユーザフィードバック
    create_feedback(run_id)

if __name__ == "__main__":
    main()
