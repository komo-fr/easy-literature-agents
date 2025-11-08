from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.prompts.rewriter import REWRITER_PROMPT


class HighlightToRewrites(BaseModel):
    highlight: str = Field(default_factory=str, description="印象的な原文フレーズ")
    rewritten: str = Field(default_factory=str, description="リライト後の文")

class RewrittenSentence(BaseModel):
    rewritten_text: str = Field(default_factory=str, description="リライトされた文章の全文")
    highlight_to_rewrites: List[HighlightToRewrites] = Field(
        default_factory=list, description="「印象的な原文フレーズ」と「リライト後の文」のペアのリスト"
    )

    def __str__(self) -> str:
        lines = []
        lines.append("---リライト後の文章---")
        lines.append(self.rewritten_text)
        lines.append("\n---各印象的な原文フレーズに対応する、リライト後の文---")

        for item in self.highlight_to_rewrites:
            lines.append(f"原文: {item.highlight}")
            lines.append(f"リライト: {item.rewritten}")
        return "\n".join(lines)


class Rewriter():
    def __init__(self, model_name: str):
        self.llm = init_chat_model(model=model_name, temperature=0.7, max_tokens=None).with_structured_output(RewrittenSentence)

    def run(self, text: str, highlight_lines: str) -> str:
        """テキストをリライトする関数"""

        prompt = ChatPromptTemplate(
            [
                ("system", REWRITER_PROMPT),
                ("human", "# 原文\n{text}\n\n# 印象的なフレーズ一覧\n{highlight_lines}"),
            ]
        )
        chain = prompt | self.llm
        rewritten_sentence : RewrittenSentence = chain.invoke({"text": text, "highlight_lines": highlight_lines})
        return rewritten_sentence
