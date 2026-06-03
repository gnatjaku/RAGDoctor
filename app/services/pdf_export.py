from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Callable

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def _safe_file_name(raw_name: str | None) -> str:
    base = (raw_name or "ragdoctor-odpowiedz").strip().lower()
    base = re.sub(r"[^a-z0-9-]+", "-", base)
    base = re.sub(r"-+", "-", base).strip("-")
    if not base:
        base = "ragdoctor-odpowiedz"
    return f"{base}.pdf"


def _chunk_url(metadata: dict | None) -> str | None:
    if not metadata:
        return None
    return metadata.get("url") or metadata.get("source_url") or metadata.get("link")


def _resolve_fonts() -> tuple[str, str]:
    # Unicode font fixes Polish diacritics in generated PDF.
    regular_name = "RAGDoctorUnicode"
    bold_name = "RAGDoctorUnicode-Bold"

    font_candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ),
    ]

    for regular_path, bold_path in font_candidates:
        if not (os.path.exists(regular_path) and os.path.exists(bold_path)):
            continue
        try:
            if regular_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(regular_name, regular_path))
            if bold_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(bold_name, bold_path))
            return regular_name, bold_name
        except Exception:
            continue

    return "Helvetica", "Helvetica-Bold"


def _break_long_token(token: str, font_name: str, font_size: float, width: float) -> list[str]:
    if not token:
        return [""]
    if pdfmetrics.stringWidth(token, font_name, font_size) <= width:
        return [token]

    chunks: list[str] = []
    current = ""
    for char in token:
        candidate = f"{current}{char}"
        if not current or pdfmetrics.stringWidth(candidate, font_name, font_size) <= width:
            current = candidate
            continue
        chunks.append(current)
        current = char

    if current:
        chunks.append(current)
    return chunks


def _wrap_text_lines(text: str, font_name: str, font_size: float, width: float) -> list[str]:
    wrapped: list[str] = []

    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            wrapped.append("")
            continue

        current_line = ""
        for token in paragraph.split():
            token_parts = _break_long_token(token, font_name, font_size, width)
            for part in token_parts:
                candidate = part if not current_line else f"{current_line} {part}"
                if pdfmetrics.stringWidth(candidate, font_name, font_size) <= width:
                    current_line = candidate
                else:
                    if current_line:
                        wrapped.append(current_line)
                    current_line = part

        if current_line:
            wrapped.append(current_line)

    return wrapped or [""]


def _draw_wrapped_text(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font_name: str,
    font_size: float,
    line_height: float,
    page_top_y: float,
    bottom_margin: float,
    on_page_break: Callable[[], None],
) -> float:
    lines = _wrap_text_lines(text, font_name, font_size, width)
    pdf.setFont(font_name, font_size)

    for line in lines:
        if y < bottom_margin + line_height:
            on_page_break()
            y = page_top_y
            pdf.setFont(font_name, font_size)
        pdf.drawString(x, y, line)
        y -= line_height

    return y


def _draw_wrapped_link(
    pdf: canvas.Canvas,
    url: str,
    x: float,
    y: float,
    width: float,
    *,
    font_name: str,
    font_size: float,
    line_height: float,
    page_top_y: float,
    bottom_margin: float,
    on_page_break: Callable[[], None],
) -> float:
    lines = _wrap_text_lines(f"URL: {url}", font_name, font_size, width)
    pdf.setFont(font_name, font_size)

    for line in lines:
        if y < bottom_margin + line_height:
            on_page_break()
            y = page_top_y
            pdf.setFont(font_name, font_size)

        text_width = pdfmetrics.stringWidth(line, font_name, font_size)
        pdf.setFillColorRGB(0.12, 0.35, 0.65)
        pdf.drawString(x, y, line)
        pdf.line(x, y - 1, x + text_width, y - 1)
        pdf.linkURL(url, (x, y - 2, x + text_width, y + line_height), relative=0)
        y -= line_height

    pdf.setFillColorRGB(0, 0, 0)
    return y


def _draw_footer(pdf: canvas.Canvas, page_width: float, margin: float, page_number: int, font_name: str) -> None:
    pdf.setFont(font_name, 9)
    pdf.setFillColorRGB(0.35, 0.35, 0.35)
    footer_text = f"Strona {page_number}"
    text_width = pdfmetrics.stringWidth(footer_text, font_name, 9)
    pdf.drawString(page_width - margin - text_width, margin - 18, footer_text)
    pdf.setFillColorRGB(0, 0, 0)


def _ensure_space(y: float, needed: float, content_bottom: float, page_top_y: float, on_page_break: Callable[[], None]) -> float:
    if y < content_bottom + needed:
        on_page_break()
        return page_top_y
    return y


def _parse_answer_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    paragraph_lines: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            blocks.append({"type": "paragraph", "text": " ".join(paragraph_lines)})
            paragraph_lines = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            blocks.append({"type": "list", "items": list_items})
            list_items = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue

        if line.startswith("* ") or line.startswith("- "):
            flush_paragraph()
            item = line[2:].strip()
            if item:
                list_items.append(item)
            continue

        flush_list()
        paragraph_lines.append(line)

    flush_paragraph()
    flush_list()
    return blocks


def _parse_bold_segments(text: str) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    idx = 0

    for match in re.finditer(r"\*\*(.+?)\*\*", text):
        start, end = match.span()
        if start > idx:
            segments.append((text[idx:start], False))
        segments.append((match.group(1), True))
        idx = end

    if idx < len(text):
        segments.append((text[idx:], False))

    return segments or [(text, False)]


def _append_line_piece(line: list[tuple[str, str]], piece: str, font_name: str) -> None:
    if not piece:
        return
    if line and line[-1][1] == font_name:
        prev_text, prev_font = line[-1]
        line[-1] = (f"{prev_text}{piece}", prev_font)
    else:
        line.append((piece, font_name))


def _wrap_rich_lines(
    segments: list[tuple[str, bool]],
    *,
    font_regular: str,
    font_bold: str,
    font_size: float,
    width: float,
) -> list[list[tuple[str, str]]]:
    lines: list[list[tuple[str, str]]] = []
    current_line: list[tuple[str, str]] = []
    current_width = 0.0

    for segment_text, is_bold in segments:
        font_name = font_bold if is_bold else font_regular
        tokens = segment_text.split()
        for token in tokens:
            token_parts = _break_long_token(token, font_name, font_size, width)
            for part in token_parts:
                prefix = " " if current_line else ""
                piece = f"{prefix}{part}"
                piece_width = pdfmetrics.stringWidth(piece, font_name, font_size)

                if current_line and (current_width + piece_width) > width:
                    lines.append(current_line)
                    current_line = [(part, font_name)]
                    current_width = pdfmetrics.stringWidth(part, font_name, font_size)
                    continue

                _append_line_piece(current_line, piece, font_name)
                current_width += piece_width

    if current_line:
        lines.append(current_line)

    return lines or [[("", font_regular)]]


def _draw_rich_text(
    pdf: canvas.Canvas,
    segments: list[tuple[str, bool]],
    x: float,
    y: float,
    width: float,
    *,
    font_regular: str,
    font_bold: str,
    font_size: float,
    line_height: float,
    page_top_y: float,
    bottom_margin: float,
    on_page_break: Callable[[], None],
) -> float:
    lines = _wrap_rich_lines(
        segments,
        font_regular=font_regular,
        font_bold=font_bold,
        font_size=font_size,
        width=width,
    )

    for line in lines:
        if y < bottom_margin + line_height:
            on_page_break()
            y = page_top_y

        cursor_x = x
        for piece, piece_font in line:
            if not piece:
                continue
            pdf.setFont(piece_font, font_size)
            pdf.drawString(cursor_x, y, piece)
            cursor_x += pdfmetrics.stringWidth(piece, piece_font, font_size)

        y -= line_height

    return y


def render_answer_pdf(answer: str, question: str | None, citations: list[dict], file_name: str | None = None) -> tuple[bytes, str]:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    font_regular, font_bold = _resolve_fonts()

    page_width, page_height = A4
    margin = 48
    content_bottom = margin + 24
    x = margin
    y = page_height - margin
    max_width = page_width - (2 * margin)
    page_number = 1

    def on_page_break() -> None:
        nonlocal page_number
        _draw_footer(pdf, page_width, margin, page_number, font_regular)
        pdf.showPage()
        page_number += 1

    pdf.setTitle("RAGDoctor - odpowiedz")
    pdf.setAuthor("RAGDoctor API")

    pdf.setFont(font_bold, 16)
    pdf.drawString(x, y, "RAGDoctor - odpowiedz")
    y -= 30

    if question and question.strip():
        y = _ensure_space(y, 42, content_bottom, page_height - margin, on_page_break)
        pdf.setFont(font_bold, 12)
        pdf.drawString(x, y, "Pytanie")
        y -= 16
        y = _draw_wrapped_text(
            pdf,
            question.strip(),
            x,
            y,
            max_width,
            font_name=font_regular,
            font_size=10,
            line_height=13,
            page_top_y=page_height - margin,
            bottom_margin=content_bottom,
            on_page_break=on_page_break,
        )
        y -= 12

    y = _ensure_space(y, 42, content_bottom, page_height - margin, on_page_break)
    pdf.setFont(font_bold, 12)
    pdf.drawString(x, y, "Odpowiedz")
    y -= 16

    answer_blocks = _parse_answer_blocks(answer.strip())
    if not answer_blocks:
        answer_blocks = [{"type": "paragraph", "text": answer.strip()}]

    for block in answer_blocks:
        if block["type"] == "paragraph":
            segments = _parse_bold_segments(block["text"])
            y = _draw_rich_text(
                pdf,
                segments,
                x,
                y,
                max_width,
                font_regular=font_regular,
                font_bold=font_bold,
                font_size=11,
                line_height=14,
                page_top_y=page_height - margin,
                bottom_margin=content_bottom,
                on_page_break=on_page_break,
            )
            y -= 6
            continue

        for item in block["items"]:
            y = _ensure_space(y, 20, content_bottom, page_height - margin, on_page_break)
            pdf.setFont(font_regular, 11)
            pdf.drawString(x, y, "-")
            y = _draw_rich_text(
                pdf,
                _parse_bold_segments(item),
                x + 12,
                y,
                max_width - 12,
                font_regular=font_regular,
                font_bold=font_bold,
                font_size=11,
                line_height=14,
                page_top_y=page_height - margin,
                bottom_margin=content_bottom,
                on_page_break=on_page_break,
            )
            y -= 2

        y -= 4

    if citations:
        y -= 14
        y = _ensure_space(y, 42, content_bottom, page_height - margin, on_page_break)
        pdf.setFont(font_bold, 12)
        pdf.drawString(x, y, "Zrodla:")
        y -= 20

        for idx, chunk in enumerate(citations, start=1):
            source_name = chunk.get("source_name") or chunk.get("chunk_id") or f"Fragment {idx}"
            score = chunk.get("score")
            score_str = f" (score: {float(score):.3f})" if isinstance(score, (float, int)) else ""

            y = _ensure_space(y, 24, content_bottom, page_height - margin, on_page_break)

            y = _draw_wrapped_text(
                pdf,
                f"{idx}. {source_name}{score_str}",
                x,
                y,
                max_width,
                font_name=font_regular,
                font_size=10,
                line_height=12,
                page_top_y=page_height - margin,
                bottom_margin=content_bottom,
                on_page_break=on_page_break,
            )
            url = _chunk_url(chunk.get("metadata") or {})
            if url:
                y = _draw_wrapped_link(
                    pdf,
                    url,
                    x + 10,
                    y,
                    max_width - 10,
                    font_name=font_regular,
                    font_size=10,
                    line_height=12,
                    page_top_y=page_height - margin,
                    bottom_margin=content_bottom,
                    on_page_break=on_page_break,
                )

            y -= 8

    _draw_footer(pdf, page_width, margin, page_number, font_regular)
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue(), _safe_file_name(file_name)

