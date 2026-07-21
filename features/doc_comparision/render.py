"""
features/doc_comparison/render.py

Renders a /compare response's segments[] two ways from the same data —
no second API call needed to switch views.
"""

import streamlit as st

# Placeholder palette — swap for the main app's real dark-theme colors
# before shipping this to real users. Kept as pale tints rather than solid
# blocks, with strikethrough reserved for removed text only, so dense
# diffs stay readable instead of turning into a wall of red/green.
COLOR_REMOVED_BG = "#3a1414"
COLOR_REMOVED_TEXT = "#ff9b9b"
COLOR_ADDED_BG = "#123a17"
COLOR_ADDED_TEXT = "#8fe3a3"

# Unchanged runs longer than this many words get collapsed to a short
# preview + a muted "N words unchanged" marker, so a diff with mostly
# untouched boilerplate (see the Form16 example) doesn't bury the actual
# changes in scrolling.
COLLAPSE_THRESHOLD_WORDS = 60
PREVIEW_WORDS = 12


def _escape(text: str) -> str:
    # Document content is untrusted input — escape before injecting into
    # unsafe_allow_html so stray '<'/'>' in a document don't break layout.
    return text.replace("<", "&lt;").replace(">", "&gt;")


def render_stats_bar(result: dict):
    stats = result["stats"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Additions", stats["additions"])
    c2.metric("Removals", stats["removals"])
    c3.metric("Pages/chunks touched", len(stats["pages_touched"]))

    position_type = result.get("position_type")
    if position_type == "chunk":
        st.caption("⚠️ One or both documents are DOCX — position numbers are text chunks, not real printed pages.")
    elif position_type == "mixed":
        st.caption("⚠️ Comparing a PDF against a DOCX — page/chunk numbers aren't directly comparable.")


def render_summary(result: dict):
    if result.get("summary"):
        with st.expander("🤖 AI summary of changes", expanded=True):
            st.caption("Generated for readability — the highlighted diff below is the exact, authoritative comparison.")
            st.write(result["summary"])


def render_legend():
    st.markdown(
        f'''<div style="display:flex;gap:16px;align-items:center;margin-bottom:12px;font-size:13px;color:#888;">
            <span style="display:flex;align-items:center;gap:6px;">
                <span style="width:10px;height:10px;border-radius:2px;background:{COLOR_REMOVED_BG};border:1px solid {COLOR_REMOVED_TEXT};"></span>Removed</span>
            <span style="display:flex;align-items:center;gap:6px;">
                <span style="width:10px;height:10px;border-radius:2px;background:{COLOR_ADDED_BG};border:1px solid {COLOR_ADDED_TEXT};"></span>Added</span>
        </div>''',
        unsafe_allow_html=True,
    )


def _page_badge(page_num) -> str:
    return (
        f'<div style="margin:18px 0 10px;">'
        f'<span style="font-size:12px;font-weight:500;background:rgba(255,255,255,0.06);'
        f'color:#aaa;padding:2px 10px;border-radius:999px;">Page/chunk {page_num}</span></div>'
    )


def _unchanged_html(text: str, show_full: bool) -> str:
    words = text.split()
    if show_full or len(words) <= COLLAPSE_THRESHOLD_WORDS:
        return f'<span>{_escape(text)}</span>'

    # Collapse: show a short lead-in, a muted count marker, then trail off.
    # This keeps large untouched boilerplate (contract preambles, form
    # legends, etc.) from dominating the view.
    preview = " ".join(words[:PREVIEW_WORDS])
    hidden_count = len(words) - PREVIEW_WORDS
    return (
        f'<span>{_escape(preview)} '
        f'<span style="color:#888;font-style:italic;"> &middot; {hidden_count} words unchanged &middot; </span></span>'
    )


def render_redline_view(segments: list, show_full_unchanged: bool = False):
    render_legend()

    html_parts = []
    last_page = None

    for seg in segments:
        page_a, page_b = seg.get("page_a"), seg.get("page_b")
        current_page = page_a if page_a is not None else page_b

        if current_page is not None and current_page != last_page:
            html_parts.append(_page_badge(current_page))
            last_page = current_page

        if seg["type"] == "unchanged":
            html_parts.append(_unchanged_html(seg["text"], show_full_unchanged))
        elif seg["type"] == "removed":
            text = _escape(seg["text"])
            html_parts.append(
                f'<span style="background:{COLOR_REMOVED_BG};color:{COLOR_REMOVED_TEXT};'
                f'text-decoration:line-through;text-decoration-thickness:1px;'
                f'border-radius:3px;padding:1px 3px;">{text}</span>'
            )
        elif seg["type"] == "added":
            text = _escape(seg["text"])
            html_parts.append(
                f'<span style="background:{COLOR_ADDED_BG};color:{COLOR_ADDED_TEXT};'
                f'border-bottom:2px solid {COLOR_ADDED_TEXT};'
                f'border-radius:3px;padding:1px 3px;">{text}</span>'
            )

    st.markdown(
        f'''<div style="background:rgba(255,255,255,0.02);border:0.5px solid rgba(255,255,255,0.08);
            border-radius:12px;padding:1.25rem 1.5rem;line-height:1.9;font-size:15px;">
            {"".join(html_parts)}</div>''',
        unsafe_allow_html=True,
    )


def render_side_by_side_view(segments: list):
    render_legend()

    left_html, right_html = [], []

    for seg in segments:
        text = _escape(seg["text"])

        if seg["type"] in ("unchanged", "removed"):
            style = "" if seg["type"] == "unchanged" else \
                f'style="background:{COLOR_REMOVED_BG};color:{COLOR_REMOVED_TEXT};text-decoration:line-through;border-radius:3px;padding:1px 3px;"'
            left_html.append(f'<span {style}>{text}</span>')

        if seg["type"] in ("unchanged", "added"):
            style = "" if seg["type"] == "unchanged" else \
                f'style="background:{COLOR_ADDED_BG};color:{COLOR_ADDED_TEXT};border-bottom:2px solid {COLOR_ADDED_TEXT};border-radius:3px;padding:1px 3px;"'
            right_html.append(f'<span {style}>{text}</span>')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02);border:0.5px solid rgba(255,255,255,0.08);'
            f'border-radius:12px;padding:1rem 1.25rem;line-height:1.9;font-size:15px;">{"".join(left_html)}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("**New version**")
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02);border:0.5px solid rgba(255,255,255,0.08);'
            f'border-radius:12px;padding:1rem 1.25rem;line-height:1.9;font-size:15px;">{"".join(right_html)}</div>',
            unsafe_allow_html=True,
        )


def render_compare_result(result: dict):
    stats = result["stats"]
    if stats["additions"] == 0 and stats["removals"] == 0:
        st.info("No differences found — the two documents are identical (after whitespace normalization).")
        return

    render_stats_bar(result)
    render_summary(result)

    view = st.radio(
        "View",
        options=["redline", "side_by_side"],
        format_func=lambda v: "Unified redline" if v == "redline" else "Side-by-side",
        horizontal=True,
        key="compare_view_toggle",
    )

    show_full_unchanged = False
    if view == "redline":
        show_full_unchanged = st.checkbox("Show full unchanged text", value=False, key="compare_show_full")

    st.divider()

    if view == "redline":
        render_redline_view(result["segments"], show_full_unchanged)
    else:
        render_side_by_side_view(result["segments"])