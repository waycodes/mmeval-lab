"""Compare HTML report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

COMPARE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>MMEvalLab Compare: {run_a} vs {run_b}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .delta {{ font-size: 1.5em; font-weight: bold; }}
        .positive {{ color: #16a34a; }}
        .negative {{ color: #dc2626; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>Run Comparison</h1>
    <p>Baseline: <code>{run_a}</code></p>
    <p>Current: <code>{run_b}</code></p>

    <h2>Overall Delta</h2>
    <div class="delta {delta_class}">{delta:+.1%}</div>
    <p>{acc_a:.1%} → {acc_b:.1%}</p>

    <h2>Example Changes</h2>
    <table>
        <tr><th>Change</th><th>Count</th></tr>
        {change_rows}
    </table>

    <h2>Slice Regressions</h2>
    <table>
        <tr><th>Slice</th><th>Baseline</th><th>Current</th><th>Delta</th></tr>
        {regression_rows}
    </table>
</body>
</html>"""


def generate_compare_report(
    run_a_id: str,
    run_b_id: str,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    change_summary: dict[str, int],
    regressions: list[tuple[str, float, float, float]],
    output_path: Path,
) -> Path:
    """Generate HTML comparison report."""
    acc_a = metrics_a.get("overall_accuracy", 0)
    acc_b = metrics_b.get("overall_accuracy", 0)
    delta = acc_b - acc_a

    change_rows = "\n".join(
        f"<tr><td>{change}</td><td>{count}</td></tr>"
        for change, count in sorted(change_summary.items())
    )

    def reg_row(name: str, base: float, curr: float, d: float) -> str:
        return f"<tr><td>{name}</td><td>{base:.1%}</td><td>{curr:.1%}</td><td>{d:+.1%}</td></tr>"

    regression_rows = "\n".join(reg_row(n, b, c, d) for n, b, c, d in regressions)

    html = COMPARE_TEMPLATE.format(
        run_a=run_a_id[:8],
        run_b=run_b_id[:8],
        acc_a=acc_a,
        acc_b=acc_b,
        delta=delta,
        delta_class="positive" if delta >= 0 else "negative",
        change_rows=change_rows,
        regression_rows=regression_rows,
    )

    output_path.write_text(html)
    return output_path
