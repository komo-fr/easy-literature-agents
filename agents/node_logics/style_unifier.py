
from langchain._core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI

from agents.prompts.style_unifier import STYLE_UNIFIER_PROMPT


class StyleUnifier():
    def __init__(self, model_name: str):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)

    def run(self, text: str, style: str) -> str:
        """文体を統一する関数"""
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                STYLE_UNIFIER_PROMPT,
            ),
            (
                "human",
                """元の文章：
{text}

統一する文体：
{style}""",
            ),
        ]
    )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"text": text, "style": style})
