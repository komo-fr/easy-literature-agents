from typing import List, Literal

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.prompts import SUGGESTER_PROMPT


class Suggetion(BaseModel):
    feedback: str = Field(default="", description="提案の元になったフィードバック")
    proposed_change: str = Field(default="", description="具体的な提案内容")
    specific_location: str = Field(default="", description="提案を適用する具体的な場所")
    priority: Literal["MUST", "Should", "May", "nits"] = Field(default="MUST", description="提案の優先度")

    def __str__(self) -> str:
        lines = []
        for name, field in Suggetion.model_fields.items():
            label = field.description or name
            value = getattr(self, name)
            lines.append(f"{label}：{value}")

        return "\n".join(lines)


class SuggesterOutput(BaseModel):
    child_based_suggestions: List[Suggetion] = Field(default_factory=list, description="小学生からのフィードバックを元にした提案")
    adult_based_suggestions: List[Suggetion] = Field(default_factory=list, description="大人からのフィードバックを元にした提案")

    def __str__(self) -> str:
        lines = []
        for name, field in SuggesterOutput.model_fields.items():
            label = field.description or name
            value = getattr(self, name)

            if isinstance(value, list):
                formatted = "\n\n".join(str(v) for v in value)
            else:
                formatted = str(value)
            lines.append(f"--- {label} ---\n{formatted}")

        return "\n\n".join(lines)

class Suggester():
    def __init__(self, model_name: str):
        self.llm = init_chat_model(model=model_name, temperature=0.3).with_structured_output(SuggesterOutput)

    def run(self, original_text,
            rewritten_text,
            child_feedback,
            adult_feedback) -> SuggesterOutput:

        """提案エージェント用のチェーンを作成する関数"""

        prompt = ChatPromptTemplate(
        [
            ("system", SUGGESTER_PROMPT),
            (
                "human",
                """元の文章：
{original_text}

リライトされた文章：
{rewritten_text}

小学生のレビューAgentからのフィードバック：
{child_feedback}

大人のレビューAgentからのフィードバック：
{adult_feedback}
""",
            ),
        ]
    )
        chain = prompt | self.llm

        return chain.invoke({"original_text": original_text,
                             "rewritten_text": rewritten_text,
                             "child_feedback": child_feedback,
                             "adult_feedback": adult_feedback})
