from typing import Any, Dict, List

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from agents.node_logics.reviewers.highlight_extractor import HighlightExtractorOutput
from agents.prompts import HIGHLIGHT_WITH_WEB_EXTRACTOR_PROMPT


class HighlightWithWebExtractorOutput(BaseModel):
    highlighted_lines_with_web: List[str] = Field(default_factory=list, description="印象的な文やセリフのリスト（Webの検索結果）")
    highlighted_lines_original_text: List[str] = Field(default_factory=list, description="印象的な文やセリフのリスト（対応する原文）")
    details: List[Dict[str, Any]] = Field(default_factory=list, description="検索した文の詳細な情報")

    def __str__(self) -> str:
        lines = []
        for name, field in HighlightWithWebExtractorOutput.model_fields.items():
            label = field.description or name
            value = getattr(self, name)
            if isinstance(value, list):
                formatted = "\n\n".join(str(v) for v in value)
            else:
                formatted = str(value)
            lines.append(f"--- {label} ---\n{formatted}")
        return "\n".join(lines)

class HighlightWithWebExtractor():
    """印象的な文やセリフを、Web検索を利用して抽出するクラス"""

    def __init__(self, model_name: str):
        self.llm = init_chat_model(model=model_name, temperature=0).with_structured_output(HighlightExtractorOutput)
        self.retriever = TavilySearchAPIRetriever(k=10)

        print("SentenceTransformerのモデルのロード中...")
        model_name = "intfloat/multilingual-e5-large"
        self.sentence_transformer_model = SentenceTransformer(model_name, device="cpu")
        print("SentenceTransformerのモデルのロード完了")

    """
    def __init__(self, model_name: str):
        self.llm = init_chat_model(model=model_name, temperature=0)
        tavily = TavilySearch(max_results=5)
        llm_with_tools = self.llm.bind_tools([tavily])
        self.llm_with_tools = llm_with_tools.with_structured_output(HighlightExtractorOutput)

        print("SentenceTransformerのモデルのロード中...")
        model_name = "intfloat/multilingual-e5-large"
        self.sentence_transformer_model = SentenceTransformer(model_name, device="cpu")
        print("SentenceTransformerのモデルのロード完了")
    """

    def _convert_original_text_to_df(self, original_text: str) -> pd.DataFrame:
        """原稿をDataFrameに変換する関数"""
        # 10個以上のハイフンが続く行を探す
        start_index = original_text.find("-"*55)
        end_index = original_text.find("-"*55, start_index + 1)
        text = original_text[end_index+55:].lstrip("\n").lstrip("\u3000")

        # テキストを1文ごとに分割する
        sentences = text.split("。")
        df = pd.DataFrame(sentences, columns=["sentence"])
        return df

    def _extract_title_and_author(self, original_text: str) -> str:
        """原稿から作者とタイトルを抽出する関数"""
        splited_text = original_text.split("\n")
        title = splited_text[0]
        author = splited_text[1]
        return title, author


    def run(self, original_text: str) -> HighlightWithWebExtractorOutput:
        """原稿から印象的な文やセリフを抽出する関数"""
        title, author = self._extract_title_and_author(original_text)

        prompt = ChatPromptTemplate([
            ("system", HIGHLIGHT_WITH_WEB_EXTRACTOR_PROMPT),
            ("human", "# 著者名: {author}\n# タイトル: {title} # コンテキスト\n{context}"),
        ])
        question_to_search = f"""{author}の「{title}」の名言、名台詞、有名なフレーズを選んでください。
        特に、SNSで有名なフレーズや、ネットミームになっているかどうかも考慮に入れてください。"""

        chain = (
        RunnablePassthrough.assign(context=(lambda x: x["question"]) | self.retriever)
        | prompt
        | self.llm
        )
        highlighted_executor_output = chain.invoke({"question": question_to_search, "author": author, "title": title})

        # 原文から類似のフレーズを探す
        targets = highlighted_executor_output.highlighted_lines
        results = self._search_similarity_sentences(original_text, targets)

        output = HighlightWithWebExtractorOutput(
            highlighted_lines_with_web=targets,
            highlighted_lines_original_text=list(set(result["similarity_sentence"] for result in results)),
            details=results  # デバッグ用で持たせる
        )

        return output

    def _search_similarity_sentence(self, target_text: str, source_df: pd.DataFrame) -> str:
        target_vector = self.sentence_transformer_model.encode([target_text])[0]

        _df = source_df.copy()
        _df["similarity"] = _df["vector"].apply(lambda x: cosine_similarity([target_vector], [x])[0])

        sim_df = _df.sort_values("similarity", ascending=False)[["sentence", "similarity"]]
        threshold = 0.85
        sim_df = sim_df[sim_df["similarity"] >= threshold]
        if sim_df.empty:
            print(target_text)
            print(f"👉 類似度が{threshold}以上のデータがありませんでした。")
            return None, None
        else:
            similarity_sentence = sim_df.iloc[0]["sentence"]
            score = sim_df.iloc[0]["similarity"][0]
            return similarity_sentence, float(score)

    def _search_similarity_sentences(self, original_text: str, targets: list[str]) -> list[float]:
        df = self._convert_original_text_to_df(original_text)
        print("原稿のベクトル化を開始...")
        df["vector"] = df["sentence"].apply(self.sentence_transformer_model.encode)
        print("原稿のベクトル化完了")

        similarity_sentences = []
        for target_text in targets:
            similarity_sentence, score = self._search_similarity_sentence(target_text, df)
            if similarity_sentence:
                data = {"target": target_text, "similarity_sentence": similarity_sentence, "score": float(score)}
                similarity_sentences.append(data)

        return similarity_sentences
