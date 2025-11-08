import uuid
from pathlib import Path
from typing import List, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from langsmith import traceable
from pydantic import BaseModel, Field

from agents.node_logics.reviewers.adult_reviewer import AdultFeedback, AdultReviewer
from agents.node_logics.reviewers.child_reviewer import ChildFeedback, ChildReviewer
from agents.node_logics.reviewers.highlight_extractor import (
    HighlightExtractor,
    HighlightExtractorOutput,
)
from agents.node_logics.reviewers.highlight_with_web_extractor import (
    HighlightWithWebExtractor,
    HighlightWithWebExtractorOutput,
)
from agents.node_logics.rewriter import Rewriter, RewrittenSentence
from agents.node_logics.style_mimic import StyleMimicAgent
from agents.node_logics.suggester import Suggester, SuggesterOutput
from agents.node_logics.suggestion_applier import SuggestionApplier, SuggestionApplierOutput


class WorkflowState(BaseModel):
    original_text: str = Field(..., description="元のテキスト")
    model: str = Field(..., description="モデル名")
    
    # 各エージェントの出力結果
    highlight_output: HighlightExtractorOutput = Field(default_factory=HighlightExtractorOutput,
                                                        description="原稿をもとに抽出した印象的な文やセリフ")
    highlight_with_web_output: HighlightWithWebExtractorOutput = Field(default_factory=HighlightWithWebExtractorOutput,
                                                        description="原稿をもとに抽出した印象的な文やセリフ")
    highlight_all: List[str] = Field(default_factory=list, description="印象的な文やセリフ")
    rewritten_sentence: RewrittenSentence = Field(default_factory=RewrittenSentence,
                                                        description="リライトされたテキストの詳細")
    child_feedback: ChildFeedback = Field(default_factory=ChildFeedback,
                                            description="小学生のレビューAgentからのフィードバック")
    adult_feedback: AdultFeedback = Field(default_factory=AdultFeedback,
                                            description="大人のレビューAgentからのフィードバック")
    suggester_output: SuggesterOutput = Field(default_factory=SuggesterOutput,
                                            description="改善提案")
    suggestion_applier_output: SuggestionApplierOutput = Field(default_factory=SuggestionApplierOutput,
                                                                description="提案を適用した後のテキストと提案の適用情報")
    style_source_text: Optional[str] = Field(default_factory=str,
                                            description="文体の模倣元となるテキスト")
    styled_text: str = Field(default_factory=str, description="模倣後のテキスト")


    def __str__(self) -> str:
        lines = []
        for name, field in WorkflowState.model_fields.items():
            label = field.description or name
            value = getattr(self, name)
            lines.append(f"====== {label} ======\n{value}")
        return "\n\n".join(lines)

class Workflow:
    def __init__(self, model_name: str):
        self.model_name = model_name

        # エージェントの初期化
        self.rewriter = Rewriter(model_name)
        self.highlight_extractor = HighlightExtractor(model_name)
        self.highlight_with_web_extractor = HighlightWithWebExtractor(model_name)
        self.child_reviewer = ChildReviewer(model_name)
        self.adult_reviewer = AdultReviewer(model_name)
        self.suggester = Suggester(model_name)
        self.suggestion_applier = SuggestionApplier(model_name)
        self.style_mimic_agent = StyleMimicAgent(model_name)

        self.graph = self._create_graph()

    @traceable(name="Litrature2Easy")
    def run(self, original_text: str, style_source_text: Optional[str] = None) -> WorkflowState:
        # 再開時に必要な識別子を作成・設定
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        state = WorkflowState(original_text=original_text,
                              model=self.model_name, style_source_text=style_source_text)
        result = self.graph.invoke(state, config=config)

        print(result["__interrupt__"][0].value["task"])

        # interrept()に到達したタイミングで処理が中断するので、現在の状態を取得
        state = WorkflowState.model_validate(result)
        edited_sugesstions = self._human_review_edit(state)

        # 残りのエージェントの処理を再開
        resumed_result = self.graph.invoke(
                        Command(resume={"edited_suggestions": edited_sugesstions}),
                        config=config)

        return resumed_result

    def _human_review_edit(self, state: WorkflowState) -> list:
        # 中断時の処理: 人間によって提案を取捨選択する
        all_edited_suggestions = []
        suggesitons = [state.suggester_output.child_based_suggestions,state.suggester_output.adult_based_suggestions]
        for suggestions in suggesitons:
            edited_suggestions = []
            n_suggestions = len(suggestions)
            for i, item in enumerate(suggestions):
                print(f"====== {i+1}/{n_suggestions} ======")
                print(item)
                print("=================")

                answer = input("この項目を残しますか？（y/n）:")
                if answer == "y":
                    edited_suggestions.append(item)
            all_edited_suggestions.append(edited_suggestions)

        edited_suggester_output = SuggesterOutput(child_based_suggestions=all_edited_suggestions[0],
        adult_based_suggestions=all_edited_suggestions[1])
        return edited_suggester_output


    def _create_graph(self):
        graph_builder = StateGraph(WorkflowState)

        # ノードの追加
        graph_builder.add_node("フレーズ抽出Agent", self._extract_highlight_node)
        graph_builder.add_node("Webからのフレーズ抽出Agent", self._extract_highlight_with_web_node)
        graph_builder.add_node("リライトAgent", self._rewrite_text_node)
        graph_builder.add_node("子供レビュアーAgent", self._review_child_node)
        graph_builder.add_node("大人レビュアーAgent", self._review_adult_node)
        graph_builder.add_node("改善提案Agent", self._suggest_improvements_node)
        graph_builder.add_node("人間による提案編集", self._human_review_edit_node)
        graph_builder.add_node("レビュー反映Agent", self._apply_suggestions_node)
        graph_builder.add_node("文体模倣Agent", self._style_mimic_agent_node)

        # エッジの追加
        graph_builder.add_edge("フレーズ抽出Agent", "Webからのフレーズ抽出Agent")
        graph_builder.add_edge("Webからのフレーズ抽出Agent", "リライトAgent")
        graph_builder.add_edge("リライトAgent", "子供レビュアーAgent")
        graph_builder.add_edge("リライトAgent", "大人レビュアーAgent")
        graph_builder.add_edge("子供レビュアーAgent", "改善提案Agent")
        graph_builder.add_edge("大人レビュアーAgent", "改善提案Agent")
        graph_builder.add_edge("改善提案Agent", "人間による提案編集")
        graph_builder.add_edge("人間による提案編集", "レビュー反映Agent")

        # 文体模倣のテキストがある場合は文体模倣を行う
        graph_builder.add_conditional_edges(
           "レビュー反映Agent",
           lambda state: state.style_source_text is not None,
           {
               True: "文体模倣Agent",
               False: END,
           },
        )

        # 開始ノードの設定
        graph_builder.set_entry_point("フレーズ抽出Agent")

        checkpointer = InMemorySaver()
        return graph_builder.compile(checkpointer=checkpointer)


    def _extract_highlight_node(self, state: WorkflowState) -> dict:
        # フレーズ抽出エージェントのノードの処理
        output = self.highlight_extractor.run(state.original_text)
        return {"highlight_output": output, "highlight_all": output.highlighted_lines}

    def _extract_highlight_with_web_node(self, state: WorkflowState) -> dict:
        # Webからのフレーズ抽出エージェントのノードの処理
        output = self.highlight_with_web_extractor.run(original_text=state.original_text)

        highlight_all = state.highlight_all
        highlight_all += output.highlighted_lines_original_text
        return {"highlight_with_web_output": output, "highlight_all": highlight_all}

    def _rewrite_text_node(self, state: WorkflowState) -> dict:
        # リライトエージェントのノードの処理
        rewritten_sentence = self.rewriter.run(state.original_text, state.highlight_all)
        return {"rewritten_sentence": rewritten_sentence}

    def _review_child_node(self, state: WorkflowState) -> dict:
        # 子供レビュアーAgentのノードの処理
        child_feedback = self.child_reviewer.run(state.rewritten_sentence.rewritten_text)
        return {"child_feedback": child_feedback}

    def _review_adult_node(self, state: WorkflowState) -> dict:
        # 大人レビュアーAgentのノードの処理
        adult_feedback = self.adult_reviewer.run(state.original_text,
                                                 state.rewritten_sentence.rewritten_text)
        return {"adult_feedback": adult_feedback}

    def _suggest_improvements_node(self, state: WorkflowState) -> dict:
        # 改善提案Agentのノードの処理
        suggester_output = self.suggester.run(state.original_text,
                                              state.rewritten_sentence.rewritten_text,
                                              state.child_feedback,
                                              state.adult_feedback)

        return {"suggester_output": suggester_output}

    def _human_review_edit_node(self, state: WorkflowState) -> dict:
        # 人間による提案編集のノードの処理
        result = interrupt({"task": "提案の取捨選択をしてください。",})
        edited_suggester_output = result["edited_suggestions"]
        return {"suggester_output": edited_suggester_output}

    def _apply_suggestions_node(self, state: WorkflowState) -> dict:
        # 提案を適用するエージェントのノードの処理
        output = self.suggestion_applier.run(state.original_text,
                                             state.rewritten_sentence.rewritten_text,
                                             state.suggester_output)
        return {"suggestion_applier_output": output}

    def _style_mimic_agent_node(self, state: WorkflowState) -> dict:
        # 文体模倣エージェントのノードの処理
        output = self.style_mimic_agent.run(state.suggestion_applier_output.revised_text, state.style_source_text)
        return {"styled_text": output.styled_text}

    def output_graph(self, path: Path = Path("images/workflow_graph.png")) -> None:
        # グラフを画像で出力
        png_data = self.graph.get_graph().draw_mermaid_png(retry_delay=2, max_retries=5)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(png_data)
