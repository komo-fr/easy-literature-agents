
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.prompts.adult_reviewer import ADULT_REVIEWER_PROMPT


class AdultFeedback(BaseModel):
    structure_consistency: str = Field(default="", description="「物語の骨子が保たれているか」という観点でのレビュー文章")
    theme_consistency: str = Field(default="", description="「作品のメッセージが保たれているか」という観点でのレビュー文章")
    symbolism_check: str = Field(default="", description="「象徴や比喩表現が適切に反映されているか」という観点でのレビュー文章")
    author_intent: str = Field(default="", description="「作者の意図が伝わっているか」という観点でのレビュー文章")

    def __str__(self) -> str:
        lines = [""]
        for name, field in AdultFeedback.model_fields.items():
            label = field.description or name
            items = getattr(self, name)
            lines.append(f"{label}：")
            lines.append(items)
            lines.append("")  # 空行

        return "\n".join(lines)

class AdultReviewer():
    def __init__(self, model_name: str):
        self.llm = init_chat_model(model=model_name, temperature=0.3).with_structured_output(AdultFeedback)

    def run(self, original_text: str, rewritten_text: str) -> AdultFeedback:
        """大人のレビュアー用のチェーンを作成する関数"""

        prompt = ChatPromptTemplate([
            ("system", ADULT_REVIEWER_PROMPT),
            (
                "human",
                """元の文章:
{original_text}

リライトされた文章：
{rewritten_text}
""",
            ),
        ])

        chain = prompt | self.llm

        return chain.invoke({"original_text": original_text,
                             "rewritten_text": rewritten_text})

