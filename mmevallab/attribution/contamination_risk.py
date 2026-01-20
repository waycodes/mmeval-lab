"""Link data diffs to contamination risk."""

from __future__ import annotations

from dataclasses import dataclass

from mmevallab.contamination.tags import ContaminationLevel, ContaminationTag


@dataclass
class ContaminationRisk:
    """Contamination risk assessment for a dataset diff."""

    added_contaminated: int
    removed_contaminated: int
    risk_level: ContaminationLevel
    message: str


def assess_diff_contamination(
    added_ids: set[str],
    removed_ids: set[str],
    contamination_tags: dict[str, ContaminationTag],
) -> ContaminationRisk:
    """Assess contamination risk from dataset diff."""
    added_contam = sum(
        1
        for eid in added_ids
        if eid in contamination_tags
        and contamination_tags[eid].level != ContaminationLevel.NONE
    )
    removed_contam = sum(
        1
        for eid in removed_ids
        if eid in contamination_tags
        and contamination_tags[eid].level != ContaminationLevel.NONE
    )

    # Determine overall risk
    if added_contam > 0:
        max_level = max(
            (
                contamination_tags[eid].level
                for eid in added_ids
                if eid in contamination_tags
            ),
            default=ContaminationLevel.NONE,
            key=lambda x: list(ContaminationLevel).index(x),
        )
    else:
        max_level = ContaminationLevel.NONE

    msg = f"Added {added_contam} contaminated, removed {removed_contam} contaminated"

    return ContaminationRisk(
        added_contaminated=added_contam,
        removed_contaminated=removed_contam,
        risk_level=max_level,
        message=msg,
    )
