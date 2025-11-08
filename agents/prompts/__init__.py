"""プロンプトモジュールの初期化ファイル"""

from .highlight_extractor import HIGHLIGHT_EXTRACTOR_PROMPT
from .suggester import SUGGESTER_PROMPT
from .suggestion_applier import SUGGESTION_APPLIER_PROMPT
from .highlight_with_web_extractor import HIGHLIGHT_WITH_WEB_EXTRACTOR_PROMPT
from .style_mimic import STYLE_MIMIC_PROMPT

__all__ = [
    "HIGHLIGHT_EXTRACTOR_PROMPT",
    "SUGGESTER_PROMPT",
    "SUGGESTION_APPLIER_PROMPT",
    "HIGHLIGHT_WITH_WEB_EXTRACTOR_PROMPT",
    "STYLE_MIMIC_PROMPT",
]
