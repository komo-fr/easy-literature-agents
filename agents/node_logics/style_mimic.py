from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.prompts import STYLE_MIMIC_PROMPT


class StyleMimicAgentOutput(BaseModel):
    """文体を模倣するエージェントの出力"""
    # style_source_text: str = Field(str, description="文体の模倣元となるテキスト")
    styled_text: str = Field(default_factory=str, description="模倣後のテキスト")

    def __str__(self) -> str:
        lines = []
        lines.append("---スタイル適用後の文章---")
        lines.append(self.styled_text)
        return "\n".join(lines)

class StyleMimicAgent():
    """文体を模倣するエージェント"""

    def __init__(self, model_name: str):
        self.llm = init_chat_model(model=model_name, temperature=0.5).with_structured_output(StyleMimicAgentOutput)

    def run(self, target_text: str, style_source_text: str) -> StyleMimicAgentOutput:
        """原稿から印象的な文やセリフを抽出する関数"""
        prompt = ChatPromptTemplate(
            [
                ("system", STYLE_MIMIC_PROMPT),
                ("human", """
                 1. 【元の文章（意味は正しいが、文体が素朴です）】
{target_text}

2. 【参考となる文体（この文体に寄せて書き直してください）】
{style_source_text}
"""),
            ]
        )

        chain = prompt | self.llm
        return chain.invoke({"target_text": target_text, "style_source_text": style_source_text})
