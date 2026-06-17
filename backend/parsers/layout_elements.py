"""
layout_elements.py — extract form-specific and visual elements from Docling output.

Three extractors, all called from DoclingParser's per-page loop:

  extract_checkboxes()       — B2: detect checked/unchecked form checkboxes
  extract_signature_regions() — B2: detect signed/unsigned signature blocks
  extract_figures()          — B3: detect figure/chart elements with bboxes

All functions:
  - Take a live docling_doc object (after conversion)
  - Return typed DocIntel model objects (LayoutElement / ImageElement)
  - Wrap all element access in try/except — a bad element never blocks ingestion
  - Return [] on any error, so calling code is never surprised
  - Normalize bounding box coordinates to [0.0, 1.0] relative to page dimensions

Design note on image_bytes:
  extract_figures() does NOT populate image_bytes on ImageElement.
  Vision description goes through describe_pdf_page() (page-level rendering)
  which provides better layout context than a raw figure crop and avoids
  bloating Supabase with binary data.
"""

from core.document import LayoutElement, ImageElement, BoundingBox
from core.logger import get_logger

logger = get_logger("layout_elements")


# ---------------------------------------------------------------------------
# Checkbox detection (B2)
# ---------------------------------------------------------------------------

# Docling label strings that map to checkbox-like elements.
# Docling's layout model emits these labels for form-heavy documents
# when the form/table-aware pipeline is active.
CHECKBOX_LABELS = {
    "checkbox_selected":   "checked",
    "checkbox_unselected": "unchecked",
}


def extract_checkboxes(
    docling_doc,
    page_num: int,
    page_size,
) -> list[LayoutElement]:
    """
    Scan Docling's element list for checkbox-type elements on this page.

    Returns LayoutElement entries with:
      element_type = 'checkbox'
      state        = 'checked' | 'unchecked'
      bbox         = normalized BoundingBox (None if page_size unavailable)

    Args:
        docling_doc: Live Docling document object after conversion.
        page_num:    1-indexed page number to filter elements by.
        page_size:   Docling page size object with .width and .height attributes.
                     Pass None if unavailable — bbox will be None on all results.

    Returns:
        List of LayoutElement (may be empty if no checkboxes found or
        if Docling model version doesn't emit checkbox labels).
    """
    results = []

    if not hasattr(docling_doc, "texts"):
        return results

    for element in docling_doc.texts:
        try:
            label = str(getattr(element, "label", "")).lower()
            if label not in CHECKBOX_LABELS:
                continue

            if not element.prov or element.prov[0].page_no != page_num:
                continue

            bbox = _extract_bbox(element.prov[0], page_num, page_size)

            results.append(LayoutElement(
                element_type="checkbox",
                page_num=page_num,
                text=getattr(element, "text", "") or "",
                confidence=0.85,
                bbox=bbox,
                state=CHECKBOX_LABELS[label],
            ))

        except Exception as e:
            logger.warning("Checkbox extraction failed for element",
                           page=page_num, error=str(e))

    return results


# ---------------------------------------------------------------------------
# Signature detection (B2)
# ---------------------------------------------------------------------------

# Text patterns that often co-occur with signature regions.
# Checked against lowercased page text — first match per page is used.
SIGNATURE_TEXT_HINTS = [
    "signature",
    "signed by",
    "authorized signatory",
    "authorised signatory",
    "sign here",
    "/s/",
    "digitally signed",
]


def extract_signature_regions(
    docling_doc,
    page_num: int,
    page_size,
    page_text: str,
) -> list[LayoutElement]:
    """
    Detect signature blocks on a page using two-signal approach.

    Signal 1 (preferred): Docling layout label 'signature' or 'handwritten_text'.
      Indicates actual signed content is present → state = 'signed'.
      Confidence: 0.75

    Signal 2 (fallback): Keyword hints in page text when no label found.
      Indicates a signature line exists, but we can't confirm if signed.
      → state = 'unknown', confidence = 0.5

    Args:
        docling_doc: Live Docling document object after conversion.
        page_num:    1-indexed page number.
        page_size:   Docling page size object (may be None).
        page_text:   Full extracted text of this page (for hint matching).

    Returns:
        List of LayoutElement with element_type='signature'.
        At most one entry per signal type. Empty if no signals found.
    """
    results = []

    # Signal 1 — explicit Docling labels
    if hasattr(docling_doc, "texts"):
        for element in docling_doc.texts:
            try:
                label = str(getattr(element, "label", "")).lower()
                if label not in ("signature", "handwritten_text"):
                    continue

                if not element.prov or element.prov[0].page_no != page_num:
                    continue

                bbox = _extract_bbox(element.prov[0], page_num, page_size)

                results.append(LayoutElement(
                    element_type="signature",
                    page_num=page_num,
                    text=getattr(element, "text", "") or "",
                    confidence=0.75,
                    bbox=bbox,
                    state="signed",
                ))

            except Exception as e:
                logger.warning("Signature label extraction failed",
                               page=page_num, error=str(e))

    # Signal 2 — text hints (only if no label-based detection found)
    if not results:
        page_text_lower = (page_text or "").lower()
        for hint in SIGNATURE_TEXT_HINTS:
            if hint in page_text_lower:
                results.append(LayoutElement(
                    element_type="signature",
                    page_num=page_num,
                    text=hint,
                    confidence=0.5,
                    bbox=None,
                    state="unknown",
                ))
                break  # one hint per page is enough

    return results


# ---------------------------------------------------------------------------
# Figure extraction (B3)
# ---------------------------------------------------------------------------

def extract_figures(
    docling_doc,
    result,
    page_num: int,
    page_size,
) -> list[ImageElement]:
    """
    Detect figure/chart elements on a page from Docling's picture list.

    Returns ImageElement entries with:
      element_type = 'figure'
      bbox         = normalized BoundingBox (None if unavailable)
      caption      = caption text from Docling if present
      image_ref    = "page_{N}_figure_{M}"
      description  = "" (filled in later by vision engine in ingestion.py)
      chunk_type   = "figure"

    NOTE: image_bytes is intentionally NOT populated. Vision description
    is generated by describe_pdf_page() in ingestion.py, which renders
    the full page at 150 DPI. This provides better layout context than
    a raw crop and keeps Supabase free of binary blobs.

    Args:
        docling_doc: Live Docling document object after conversion.
        result:      Full Docling conversion result (used to access page images
                     if generate_page_images=True; currently unused since we
                     skip image_bytes, kept for API compatibility).
        page_num:    1-indexed page number.
        page_size:   Docling page size object (may be None).

    Returns:
        List of ImageElement. Empty if no figures detected or docling_doc
        has no .pictures attribute.
    """
    images = []

    if not hasattr(docling_doc, "pictures"):
        return images

    figure_index = 0
    for picture in docling_doc.pictures:
        try:
            if not picture.prov or picture.prov[0].page_no != page_num:
                continue

            bbox = _extract_bbox(picture.prov[0], page_num, page_size)
            caption = getattr(picture, "caption", "") or ""

            # caption may be a Docling caption object rather than a plain string
            if not isinstance(caption, str):
                caption = str(getattr(caption, "text", ""))

            figure_index += 1
            image_ref = f"page_{page_num}_figure_{figure_index}"

            images.append(ImageElement(
                page_num=page_num,
                image_ref=image_ref,
                ocr_text="",
                description="",            # filled by vision engine in ingestion.py
                chunk_type="figure",
                vision_prompt_used="",     # set when vision runs
                bbox=bbox,
                caption=caption,
                element_type="figure",
            ))

        except Exception as e:
            logger.warning("Figure extraction failed",
                           page=page_num, error=str(e))

    return images


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_bbox(prov, page_num: int, page_size) -> "BoundingBox | None":
    """
    Normalize a Docling provenance bbox to a BoundingBox.

    Docling bbox coordinates (l, t, r, b) are in points (absolute).
    We normalize to [0.0, 1.0] relative to page dimensions.

    Returns None if:
      - page_size is None or has zero dimensions
      - prov.bbox is unavailable
      - Any arithmetic error occurs (e.g. ZeroDivisionError)
    """
    try:
        if not page_size or not page_size.width or not page_size.height:
            return None

        bbox_raw = prov.bbox
        w = page_size.width
        h = page_size.height

        return BoundingBox(
            x=bbox_raw.l / w,
            y=bbox_raw.t / h,
            width=(bbox_raw.r - bbox_raw.l) / w,
            height=(bbox_raw.b - bbox_raw.t) / h,
            page_num=page_num,
            page_width=w,
            page_height=h,
        )

    except Exception as e:
        logger.debug("BoundingBox extraction failed", page=page_num, error=str(e))
        return None