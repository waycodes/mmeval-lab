"""PDF page rendering with caching."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def _compute_pdf_hash(pdf_path: Path) -> str:
    """Compute hash of PDF file for cache key."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _get_cache_key(pdf_path: Path, page_num: int, dpi: int) -> str:
    """Generate cache key for rendered page."""
    pdf_hash = _compute_pdf_hash(pdf_path)
    return f"{pdf_hash}_p{page_num}_dpi{dpi}"


class PDFRenderer:
    """Cacheable PDF page renderer using PyMuPDF."""

    def __init__(self, cache_dir: Path | str | None = None, dpi: int = 144) -> None:
        """Initialize renderer.

        Args:
            cache_dir: Directory for caching rendered images (None = no caching)
            dpi: Resolution for rendering (default 144)
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._dpi = dpi

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def render_page(
        self,
        pdf_path: Path | str,
        page_num: int,
        dpi: int | None = None,
    ) -> "Image.Image":
        """Render a PDF page to PIL Image.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            dpi: Override default DPI

        Returns:
            PIL Image of rendered page
        """
        try:
            import fitz  # PyMuPDF
            from PIL import Image
        except ImportError as e:
            raise ImportError("Install pymupdf and pillow: pip install pymupdf pillow") from e

        pdf_path = Path(pdf_path)
        dpi = dpi or self._dpi

        # Check cache
        if self._cache_dir:
            cache_key = _get_cache_key(pdf_path, page_num, dpi)
            cache_path = self._cache_dir / f"{cache_key}.png"
            if cache_path.exists():
                return Image.open(cache_path)

        # Render page
        doc = fitz.open(pdf_path)
        try:
            # PyMuPDF uses 0-indexed pages
            page = doc[page_num - 1]
            zoom = dpi / 72.0  # PDF default is 72 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()

        # Cache result
        if self._cache_dir:
            img.save(cache_path, "PNG")

        return img

    def render_page_bytes(
        self,
        pdf_path: Path | str,
        page_num: int,
        dpi: int | None = None,
        format: str = "PNG",
    ) -> bytes:
        """Render a PDF page to bytes.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            dpi: Override default DPI
            format: Image format (PNG, JPEG)

        Returns:
            Image bytes
        """
        from io import BytesIO

        img = self.render_page(pdf_path, page_num, dpi)
        buf = BytesIO()
        img.save(buf, format=format)
        return buf.getvalue()

    def clear_cache(self) -> int:
        """Clear all cached renders.

        Returns:
            Number of files removed
        """
        if not self._cache_dir:
            return 0

        count = 0
        for f in self._cache_dir.glob("*.png"):
            f.unlink()
            count += 1
        return count


# Default renderer instance
_default_renderer: PDFRenderer | None = None


def get_renderer(cache_dir: Path | str | None = None, dpi: int = 144) -> PDFRenderer:
    """Get or create default PDF renderer."""
    global _default_renderer
    if _default_renderer is None:
        _default_renderer = PDFRenderer(cache_dir=cache_dir, dpi=dpi)
    return _default_renderer


def render_pdf_page(
    pdf_path: Path | str,
    page_num: int,
    dpi: int = 144,
    cache_dir: Path | str | None = None,
) -> "Image.Image":
    """Convenience function to render a PDF page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        dpi: Resolution
        cache_dir: Cache directory (None = no caching)

    Returns:
        PIL Image
    """
    renderer = PDFRenderer(cache_dir=cache_dir, dpi=dpi)
    return renderer.render_page(pdf_path, page_num, dpi)
