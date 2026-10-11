"""Write the report as one self-contained HTML file (no internet needed to open it)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEMPLATE = Path(__file__).with_name("template.html")
_PLACEHOLDER = "/*__GTS_REPORT_DATA__*/null"


def render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = TEMPLATE.read_text(encoding="utf-8")
    if _PLACEHOLDER not in template:
        raise RuntimeError("report template has no data placeholder")
    return template.replace(_PLACEHOLDER, payload, 1)


def write_report(data: dict[str, Any], html_path: Path, json_path: Path | None = None) -> None:
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(data), encoding="utf-8")
    if json_path is not None:
        Path(json_path).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
