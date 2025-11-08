from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.prompts import HIGHLIGHT_EXTRACTOR_PROMPT


class HighlightExtractorOutput(BaseModel):
    highlighted_lines: List[str] = Field(default_factory=list, description="印象的な文やセリフのリスト（最大5件程度）")

    def __str__(self) -> str:
        lines = []
        for name, field in HighlightExtractorOutput.model_fields.items():
            label = field.description or name
            value = getattr(self, name)
            if isinstance(value, list):
                formatted = "\n\n".join(str(v) for v in value)
            else:
                formatted = str(value)
            lines.append(f"--- {label} ---\n{formatted}")
        return "\n".join(lines)


class HighlightExtractor():
    """印象的な文やセリフを抽出するクラス"""

    def __init__(self, model_name: str):
        self.llm = init_chat_model(model=model_name, temperature=0.5).with_structured_output(HighlightExtractorOutput)

    def run(self, text: str) -> HighlightExtractorOutput:
        """原稿から印象的な文やセリフを抽出する関数"""
        prompt = ChatPromptTemplate([
                ("system", HIGHLIGHT_EXTRACTOR_PROMPT),
                ("human", "{text}"),
            ])
        chain = prompt | self.llm
        return chain.invoke({"text": text})
