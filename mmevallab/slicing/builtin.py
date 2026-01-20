"""Built-in slice definitions for benchmarks."""

from mmevallab.slicing.engine import SliceSpec

# MMMU slices by discipline
MMMU_DISCIPLINE_SLICES = [
    SliceSpec(name="discipline:art_design", field="discipline", value="Art & Design"),
    SliceSpec(name="discipline:business", field="discipline", value="Business"),
    SliceSpec(name="discipline:health_medicine", field="discipline", value="Health & Medicine"),
    SliceSpec(name="discipline:humanities", field="discipline", value="Humanities & Social"),
    SliceSpec(name="discipline:science", field="discipline", value="Science"),
    SliceSpec(name="discipline:tech_engineering", field="discipline", value="Tech & Engineering"),
]

# MMMU slices by image type
MMMU_IMAGE_TYPE_SLICES = [
    SliceSpec(name="image_type:diagram", field="image_type", op="contains", value="diagram"),
    SliceSpec(name="image_type:chart", field="image_type", op="contains", value="chart"),
    SliceSpec(name="image_type:photo", field="image_type", op="contains", value="photo"),
    SliceSpec(name="image_type:table", field="image_type", op="contains", value="table"),
]

# MMMU slices by number of images
MMMU_NUM_IMAGES_SLICES = [
    SliceSpec(name="num_images:1", field="num_images", value=1),
    SliceSpec(name="num_images:2+", field="num_images", op="gte", value=2),
]

MMMU_SLICES = MMMU_DISCIPLINE_SLICES + MMMU_IMAGE_TYPE_SLICES + MMMU_NUM_IMAGES_SLICES

# OmniDocBench slices
OMNIDOCBENCH_DOC_TYPE_SLICES = [
    SliceSpec(name="doc_type:academic", field="doc_type", value="academic"),
    SliceSpec(name="doc_type:financial", field="doc_type", value="financial"),
    SliceSpec(name="doc_type:textbook", field="doc_type", value="textbook"),
    SliceSpec(name="doc_type:form", field="doc_type", value="form"),
]

OMNIDOCBENCH_LAYOUT_SLICES = [
    SliceSpec(name="layout:single_column", field="layout_type", value="single_column"),
    SliceSpec(name="layout:multi_column", field="layout_type", value="multi_column"),
]

OMNIDOCBENCH_CONTENT_SLICES = [
    SliceSpec(name="has_formula:true", field="has_formula", value=True),
    SliceSpec(name="has_table:true", field="has_table", value=True),
]

OMNIDOCBENCH_SLICES = (
    OMNIDOCBENCH_DOC_TYPE_SLICES + OMNIDOCBENCH_LAYOUT_SLICES + OMNIDOCBENCH_CONTENT_SLICES
)

# Video-MME slices
VIDEOMME_DURATION_SLICES = [
    SliceSpec(name="duration:short", field="duration", value="short"),
    SliceSpec(name="duration:medium", field="duration", value="medium"),
    SliceSpec(name="duration:long", field="duration", value="long"),
]

VIDEOMME_DOMAIN_SLICES = [
    SliceSpec(name="domain:knowledge", field="domain", op="contains", value="Knowledge"),
    SliceSpec(name="domain:film", field="domain", op="contains", value="Film"),
    SliceSpec(name="domain:sports", field="domain", op="contains", value="Sports"),
    SliceSpec(name="domain:life", field="domain", op="contains", value="Life"),
]

VIDEOMME_TASK_TYPE_SLICES = [
    SliceSpec(name="task:perception", field="task_type", value="perception"),
    SliceSpec(name="task:reasoning", field="task_type", value="reasoning"),
    SliceSpec(name="task:temporal", field="task_type", value="temporal"),
]

VIDEOMME_SLICES = VIDEOMME_DURATION_SLICES + VIDEOMME_DOMAIN_SLICES + VIDEOMME_TASK_TYPE_SLICES


def get_benchmark_slices(benchmark: str) -> list[SliceSpec]:
    """Get built-in slices for a benchmark."""
    slices_map = {
        "mmmu": MMMU_SLICES,
        "omnidocbench": OMNIDOCBENCH_SLICES,
        "videomme": VIDEOMME_SLICES,
    }
    return slices_map.get(benchmark, [])
