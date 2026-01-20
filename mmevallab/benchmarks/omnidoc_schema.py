"""OmniDocBench JSON schema parser and validator."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_list(cls, coords: list[float]) -> "BoundingBox":
        """Create from [x, y, w, h] list."""
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
        return cls(x=coords[0], y=coords[1], width=coords[2], height=coords[3])


class LayoutElement(BaseModel):
    """A layout element in a document page."""

    type: str = Field(description="Element type: text, title, table, figure, formula, etc.")
    bbox: BoundingBox | None = None
    content: str | None = None
    confidence: float | None = None


class OmniDocAnnotation(BaseModel):
    """Annotation for a single document page in OmniDocBench."""

    doc_id: str
    page_num: int = Field(ge=1)
    pdf_name: str
    target: str | None = Field(default=None, description="Ground truth output")

    # Document metadata
    doc_type: str = Field(default="unknown")
    layout_type: str = Field(default="unknown")
    language: str = Field(default="en")

    # Content flags
    has_formula: bool = False
    has_table: bool = False

    # Layout elements (optional)
    elements: list[LayoutElement] = Field(default_factory=list)

    # Task-specific fields
    bbox: list[float] | None = Field(default=None, description="ROI for formula/table tasks")

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, v: Any) -> list[float] | None:
        if v is None:
            return None
        if isinstance(v, list) and len(v) == 4:
            return [float(x) for x in v]
        raise ValueError("bbox must be [x, y, width, height]")


class OmniDocDataset(BaseModel):
    """Full OmniDocBench dataset schema."""

    version: str = "1.0"
    split: str
    annotations: list[OmniDocAnnotation]

    @property
    def num_examples(self) -> int:
        return len(self.annotations)


def parse_omnidoc_annotation(data: dict[str, Any]) -> OmniDocAnnotation:
    """Parse a single annotation dict into OmniDocAnnotation.

    Args:
        data: Raw annotation dictionary

    Returns:
        Validated OmniDocAnnotation

    Raises:
        ValueError: If required fields are missing or invalid
    """
    return OmniDocAnnotation.model_validate(data)


def parse_omnidoc_dataset(data: list[dict[str, Any]], split: str = "test") -> OmniDocDataset:
    """Parse full dataset annotations.

    Args:
        data: List of annotation dictionaries
        split: Dataset split name

    Returns:
        Validated OmniDocDataset
    """
    annotations = [parse_omnidoc_annotation(item) for item in data]
    return OmniDocDataset(split=split, annotations=annotations)


def validate_omnidoc_schema(data: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
    """Validate OmniDocBench data against schema.

    Args:
        data: Single annotation or list of annotations

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if isinstance(data, dict):
        data = [data]

    required_keys = {"doc_id", "page_num", "pdf_name"}

    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            errors.append(f"Item {i}: missing required keys: {missing}")

        if "page_num" in item:
            try:
                page = int(item["page_num"])
                if page < 1:
                    errors.append(f"Item {i}: page_num must be >= 1, got {page}")
            except (TypeError, ValueError):
                errors.append(f"Item {i}: page_num must be integer")

        if "bbox" in item and item["bbox"] is not None:
            bbox = item["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"Item {i}: bbox must be [x, y, width, height]")

    return errors
