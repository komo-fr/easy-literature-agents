from typing import List, Literal

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.prompts import SUGGESTION_APPLIER_PROMPT


class SuggestionDecision(BaseModel):
    source_group: Literal["child_based_suggestions", "adult_based_suggestions"] = Field(default="", description="提案の種類")
    suggestion_content: str = Field(default="", description="提案の内容")
    decision_reason: str = Field(default="", description="提案の理由")
    applied_text: str = Field(default="", description="提案を適用した後の文章")

    def __str__(self) -> str:
        lines = []
        for name, field in SuggestionDecision.model_fields.items():
            label = field.description or name
            value = getattr(self, name)
            lines.append(f"{label}: {value}")
        return "\n".join(lines)


class SuggestionApplierOutput(BaseModel):
    revised_text: str = Field(default="", description="提案を適用した後の文章")
    adopted_suggestions: List[SuggestionDecision] = Field(default_factory=list, description="採用した提案のリスト")
    rejected_suggestions: List[SuggestionDecision] = Field(default_factory=list, description="不採用にした提案のリスト")

    def __str__(self) -> str:
        lines = []
        for name, field in SuggestionApplierOutput.model_fields.items():
            label = field.description or name
            value = getattr(self, name)

            if isinstance(value, list):
                formatted = "\n\n".join(str(v) for v in value)
            else:
                formatted = str(value)
            lines.append(f"--- {label} ---\n{formatted}")

        return "\n\n".join(lines)


class SuggestionApplier():
    def __init__(self, model_name: str):
        self.llm = init_chat_model(model=model_name, temperature=0.3).with_structured_output(SuggestionApplierOutput)

    def run(self, original_text, rewritten_text, suggester_output) -> SuggestionApplierOutput:
        """提案の適用を実行する関数"""

        prompt = ChatPromptTemplate(
            [
                ("system", SUGGESTION_APPLIER_PROMPT),
                ("human",
                """元の文章（参考用）：
{original_text}

リライトされた文章（修正対象）：
{rewritten_text}

【大人視点の改善提案】：
{adult_based_suggestions}

【子ども視点の改善提案】：
{child_based_suggestions}""",
            ),
        ]
    )
        chain = prompt | self.llm

        return chain.invoke({"original_text": original_text,
                             "rewritten_text": rewritten_text,
                             "child_based_suggestions": suggester_output.child_based_suggestions,
                             "adult_based_suggestions": suggester_output.adult_based_suggestions})
