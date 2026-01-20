"""HTML report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>MMEvalLab Report: {title}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2563eb; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; }}
        .good {{ color: #16a34a; }}
        .bad {{ color: #dc2626; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Run ID: <code>{run_id}</code></p>
    <p>Generated: {timestamp}</p>

    <h2>Overall Metrics</h2>
    <div class="metric">
        <div class="metric-value">{accuracy:.1%}</div>
        <div>Accuracy ({correct}/{total})</div>
    </div>

    <h2>Breakdown by Category</h2>
    <table>
        <tr><th>Category</th><th>Accuracy</th><th>Count</th></tr>
        {breakdown_rows}
    </table>

    <h2>Worst Slices</h2>
    <table>
        <tr><th>Slice</th><th>Accuracy</th><th>Count</th></tr>
        {worst_rows}
    </table>
</body>
</html>"""


def generate_run_report(
    run_id: str,
    metrics: dict[str, Any],
    breakdown: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate HTML report for a single run."""
    from datetime import datetime

    def row(cat: str, data: dict[str, Any], cls: str = "") -> str:
        acc = data.get("accuracy", 0)
        tot = data.get("total", 0)
        td_cls = f" class='{cls}'" if cls else ""
        return f"<tr><td>{cat}</td><td{td_cls}>{acc:.1%}</td><td>{tot}</td></tr>"

    breakdown_rows = "\n".join(row(c, d) for c, d in sorted(breakdown.items()))

    worst = sorted(breakdown.items(), key=lambda x: x[1].get("accuracy", 0))[:10]
    worst_rows = "\n".join(row(c, d, "bad") for c, d in worst)

    html = HTML_TEMPLATE.format(
        title=f"Run {run_id[:8]}",
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        accuracy=metrics.get("overall_accuracy", 0),
        correct=metrics.get("correct", 0),
        total=metrics.get("total", 0),
        breakdown_rows=breakdown_rows,
        worst_rows=worst_rows,
    )

    output_path.write_text(html)
    return output_path
