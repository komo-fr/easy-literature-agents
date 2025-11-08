from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.prompts.child_reviewer import CHILD_REVIEWER_PROMPT


class ChildFeedback(BaseModel):
    difficult_words: List[str] = Field(default_factory=list, description="むずかしい言葉のリスト")
    long_sentences: List[str] = Field(default_factory=list, description="長すぎる文のリスト")
    unclear_parts: List[str] = Field(default_factory=list, description="わかりにくい部分のリスト")

    def __str__(self) -> str:
        lines = []
        for name, field in ChildFeedback.model_fields.items():
            label = field.description or name
            items = getattr(self, name)
            lines.append(f"{label}：")
            if items:
                lines.append(", ".join(items))
            lines.append("")  # 空行
        return "\n".join(lines)

class ChildReviewer():
    def __init__(self, model_name: str):
        self.llm = init_chat_model(model=model_name, temperature=0.3, max_tokens=4000).with_structured_output(ChildFeedback)

    def run(self, text: str) -> ChildFeedback:
        """小学生のレビュアー用のチェーンを作成する関数"""

        prompt = ChatPromptTemplate([
            ("system", CHILD_REVIEWER_PROMPT,),
            ("human", "{text}"),
        ])
        chain = prompt | self.llm
        return chain.invoke({"text": text})
