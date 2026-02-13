#!/usr/bin/env python3
"""Render and compare HWPX documents for quality validation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from shutil import which
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageChops, ImageStat

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

PDF_PAGE_SIZE_RE = re.compile(r"^\s*Page size:\s*([\d.]+)\s*x\s*([\d.]+)\s*pts")
PDF_PAGES_RE = re.compile(r"^\s*Pages:\s*(\d+)\s*$")
EXPORT_ENV_NAME = "HANCOM_PDF_EXPORT_CMD"


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    if debug:
        # TODO: Remove this debug breadcrumb after enough production runs.
        logging.debug("Debug logging enabled")


def ensure_render_tools(require_text_tools: bool = False) -> None:
    missing: List[str] = []
    if which("pdftoppm") is None:
        missing.append("pdftoppm")
    if which("pdfinfo") is None:
        missing.append("pdfinfo")
    if require_text_tools and which("pdftotext") is None:
        missing.append("pdftotext")

    if missing:
        tool_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing required tool(s): {tool_list}. Install Poppler tools, then retry."
        )


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    logging.debug("Running command: %s", " ".join(cmd))
    process = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        stderr = (process.stderr or "").strip()
        stdout = (process.stdout or "").strip()
        details = stderr or stdout or "No process output"
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{details}")
    return process


def build_export_command(template: str, input_path: Path, output_pdf: Path) -> List[str]:
    if "{input}" not in template or "{output}" not in template:
        raise ValueError("Export command template must include {input} and {output} placeholders")

    rendered = template.format(input=str(input_path), output=str(output_pdf))
    parts = shlex.split(rendered)
    if not parts:
        raise ValueError("Export command template produced an empty command")
    return parts


def export_document_to_pdf(input_path: Path, output_pdf: Path, export_template: str) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input is not a file: {input_path}")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_export_command(export_template, input_path, output_pdf)
    process = run_cmd(cmd)

    if output_pdf.exists():
        return output_pdf

    # Why: some exporters rewrite output filename instead of requested one.
    candidates = sorted(output_pdf.parent.glob("*.pdf"))
    if len(candidates) == 1:
        logging.debug("Expected output missing, using fallback PDF candidate: %s", candidates[0])
        return candidates[0]

    stderr = (process.stderr or "").strip()
    stdout = (process.stdout or "").strip()
    details: List[str] = []
    if stderr:
        details.append(f"stderr={stderr}")
    if stdout:
        details.append(f"stdout={stdout}")
    if candidates:
        details.append(f"pdf_candidates={[str(item) for item in candidates]}")
    detail_text = "; ".join(details) if details else "No converter output available"
    raise RuntimeError(f"PDF output missing after export: {output_pdf}. Details: {detail_text}")


def resolve_pdf_source(
    input_path: Path,
    explicit_pdf_path: Optional[Path],
    output_pdf_path: Path,
    export_template: Optional[str],
) -> Tuple[Path, str]:
    if explicit_pdf_path:
        if not explicit_pdf_path.exists():
            raise FileNotFoundError(f"PDF path not found: {explicit_pdf_path}")
        if not explicit_pdf_path.is_file():
            raise ValueError(f"PDF path is not a file: {explicit_pdf_path}")
        if explicit_pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF path must point to a .pdf file: {explicit_pdf_path}")
        return explicit_pdf_path, "explicit_pdf"

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input is not a file: {input_path}")

    if input_path.suffix.lower() == ".pdf":
        return input_path, "input_pdf"

    if not export_template:
        raise RuntimeError(
            "No export command configured. Provide --pdf-path or set --export-cmd / "
            f"{EXPORT_ENV_NAME} with {{input}} and {{output}} placeholders."
        )

    exported_pdf = export_document_to_pdf(input_path, output_pdf_path, export_template)
    return exported_pdf, "export_cmd"


def parse_pdfinfo(pdf_path: Path) -> Dict[str, object]:
    process = run_cmd(["pdfinfo", str(pdf_path)])
    pages: Optional[int] = None
    width_pts: Optional[float] = None
    height_pts: Optional[float] = None

    for line in process.stdout.splitlines():
        page_match = PDF_PAGES_RE.match(line)
        if page_match:
            pages = int(page_match.group(1))
            continue

        size_match = PDF_PAGE_SIZE_RE.match(line)
        if size_match:
            width_pts = float(size_match.group(1))
            height_pts = float(size_match.group(2))

    if pages is None:
        raise RuntimeError(f"Could not parse page count from pdfinfo: {pdf_path}")
    if width_pts is None or height_pts is None:
        raise RuntimeError(f"Could not parse page size from pdfinfo: {pdf_path}")

    return {
        "pages": pages,
        "width_pts": width_pts,
        "height_pts": height_pts,
    }


def render_pdf_to_png(pdf_path: Path, output_dir: Path, prefix: str, dpi: int) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = output_dir / prefix
    cmd = [
        "pdftoppm",
        "-r",
        str(dpi),
        "-png",
        str(pdf_path),
        str(prefix_path),
    ]
    run_cmd(cmd)

    rendered = sorted(output_dir.glob(f"{prefix}-*.png"))
    if not rendered:
        raise RuntimeError(f"No rendered pages produced for: {pdf_path}")
    return rendered


def extract_pdf_text(pdf_path: Path) -> str:
    process = run_cmd(["pdftotext", "-layout", str(pdf_path), "-"])
    return process.stdout


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def text_similarity(before_text: str, after_text: str) -> float:
    return SequenceMatcher(None, before_text, after_text).ratio()


def compare_pdf_info(
    before_info: Dict[str, object],
    after_info: Dict[str, object],
) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    before_pages = int(before_info["pages"])
    after_pages = int(after_info["pages"])
    if before_pages != after_pages:
        warnings.append(f"Page count changed: {before_pages} -> {after_pages}")

    before_size = (float(before_info["width_pts"]), float(before_info["height_pts"]))
    after_size = (float(after_info["width_pts"]), float(after_info["height_pts"]))
    # Why: allow tiny float noise from conversion pipeline.
    size_tolerance = 0.5
    if abs(before_size[0] - after_size[0]) > size_tolerance or abs(
        before_size[1] - after_size[1]
    ) > size_tolerance:
        warnings.append(
            f"Page size changed: {before_size[0]}x{before_size[1]} -> {after_size[0]}x{after_size[1]}"
        )

    return len(warnings) == 0, warnings


def page_diff_ratio(before_page: Path, after_page: Path) -> float:
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for page image diff comparison")

    with Image.open(before_page) as before_img, Image.open(after_page) as after_img:
        before_rgb = before_img.convert("RGB")
        after_rgb = after_img.convert("RGB")

        if before_rgb.size != after_rgb.size:
            return 1.0

        diff = ImageChops.difference(before_rgb, after_rgb)
        stat = ImageStat.Stat(diff)
        mean = sum(stat.mean) / max(len(stat.mean), 1)
        return mean / 255.0


def compare_rendered_pages(
    before_pages: List[Path],
    after_pages: List[Path],
    max_page_diff: float,
) -> Tuple[List[str], Dict[str, object]]:
    warnings: List[str] = []
    metrics: Dict[str, object] = {
        "pil_available": PIL_AVAILABLE,
        "compared_pages": 0,
        "max_diff_ratio": None,
        "avg_diff_ratio": None,
        "per_page_diff_ratio": [],
    }

    if len(before_pages) != len(after_pages):
        warnings.append(
            f"Rendered image count changed: {len(before_pages)} -> {len(after_pages)}"
        )

    if not PIL_AVAILABLE:
        warnings.append("Pillow unavailable: skipped pixel diff checks")
        return warnings, metrics

    compare_count = min(len(before_pages), len(after_pages))
    per_page: List[float] = []

    for index in range(compare_count):
        ratio = page_diff_ratio(before_pages[index], after_pages[index])
        per_page.append(ratio)
        if ratio > max_page_diff:
            warnings.append(
                f"Page {index + 1} diff ratio {ratio:.4f} exceeds threshold {max_page_diff:.4f}"
            )

    if per_page:
        metrics["compared_pages"] = compare_count
        metrics["max_diff_ratio"] = max(per_page)
        metrics["avg_diff_ratio"] = sum(per_page) / len(per_page)
        metrics["per_page_diff_ratio"] = [round(value, 6) for value in per_page]

    return warnings, metrics


def command_render(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    explicit_pdf = Path(args.pdf_path).expanduser().resolve() if args.pdf_path else None
    export_template = args.export_cmd or os.getenv(EXPORT_ENV_NAME)

    ensure_render_tools(require_text_tools=False)
    pdf_dir = output_dir / "pdf"
    png_dir = output_dir / "png"

    pdf_path, conversion_mode = resolve_pdf_source(
        input_path=input_path,
        explicit_pdf_path=explicit_pdf,
        output_pdf_path=pdf_dir / f"{input_path.stem}.pdf",
        export_template=export_template,
    )
    pages = render_pdf_to_png(pdf_path, png_dir, "page", args.dpi)
    pdf_meta = parse_pdfinfo(pdf_path)

    result = {
        "input": str(input_path),
        "pdf_path": str(pdf_path),
        "conversion_mode": conversion_mode,
        "rendered_pages": len(pages),
        "png_dir": str(png_dir),
        "page_meta": pdf_meta,
    }

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_output:
        json_path = Path(args.json_output).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    before_path = Path(args.before).expanduser().resolve()
    after_path = Path(args.after).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    before_pdf_explicit = (
        Path(args.before_pdf).expanduser().resolve() if args.before_pdf else None
    )
    after_pdf_explicit = (
        Path(args.after_pdf).expanduser().resolve() if args.after_pdf else None
    )
    export_template = args.export_cmd or os.getenv(EXPORT_ENV_NAME)

    ensure_render_tools(require_text_tools=True)
    before_pdf_dir = output_dir / "before" / "pdf"
    after_pdf_dir = output_dir / "after" / "pdf"
    before_png_dir = output_dir / "before" / "png"
    after_png_dir = output_dir / "after" / "png"

    before_pdf, before_mode = resolve_pdf_source(
        input_path=before_path,
        explicit_pdf_path=before_pdf_explicit,
        output_pdf_path=before_pdf_dir / f"{before_path.stem}.pdf",
        export_template=export_template,
    )
    after_pdf, after_mode = resolve_pdf_source(
        input_path=after_path,
        explicit_pdf_path=after_pdf_explicit,
        output_pdf_path=after_pdf_dir / f"{after_path.stem}.pdf",
        export_template=export_template,
    )

    before_pages = render_pdf_to_png(before_pdf, before_png_dir, "page", args.dpi)
    after_pages = render_pdf_to_png(after_pdf, after_png_dir, "page", args.dpi)

    before_info = parse_pdfinfo(before_pdf)
    after_info = parse_pdfinfo(after_pdf)
    info_ok, info_warnings = compare_pdf_info(before_info, after_info)

    before_text = normalize_text(extract_pdf_text(before_pdf))
    after_text = normalize_text(extract_pdf_text(after_pdf))
    similarity = text_similarity(before_text, after_text)

    warnings: List[str] = list(info_warnings)
    if similarity < args.min_text_similarity:
        warnings.append(
            f"Text similarity {similarity:.4f} below threshold {args.min_text_similarity:.4f}"
        )
    page_warnings, page_metrics = compare_rendered_pages(
        before_pages,
        after_pages,
        args.max_page_diff,
    )
    warnings.extend(page_warnings)

    status = "pass" if not warnings and info_ok else "fail"
    result = {
        "before": str(before_path),
        "after": str(after_path),
        "before_pdf": str(before_pdf),
        "after_pdf": str(after_pdf),
        "before_conversion_mode": before_mode,
        "after_conversion_mode": after_mode,
        "status": status,
        "similarity": round(similarity, 6),
        "min_text_similarity": args.min_text_similarity,
        "before_pages": len(before_pages),
        "after_pages": len(after_pages),
        "before_meta": before_info,
        "after_meta": after_info,
        "page_diff_metrics": page_metrics,
        "max_page_diff": args.max_page_diff,
        "warnings": warnings,
        "artifacts_dir": str(output_dir),
    }

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_output:
        json_path = Path(args.json_output).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    if args.strict and warnings:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render HWPX/PDF and compare quality with PDF-based gates."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser(
        "render",
        help="Render document (or PDF) to PNG pages",
    )
    render_parser.add_argument("input", help="Path to input .hwpx/.hwp/.pdf")
    render_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store PDF and PNG artifacts",
    )
    render_parser.add_argument(
        "--pdf-path",
        help="Existing PDF path exported by Hancom Office (skip conversion command)",
    )
    render_parser.add_argument(
        "--export-cmd",
        help=(
            "Export command template with {input} and {output} placeholders. "
            f"If omitted, {EXPORT_ENV_NAME} is used."
        ),
    )
    render_parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Render DPI for PNG output (default: 180)",
    )
    render_parser.add_argument(
        "--json-output",
        help="Optional JSON summary output path",
    )
    render_parser.set_defaults(func=command_render)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare before/after via rendered PDF metadata, text, and pixel diff",
    )
    compare_parser.add_argument("before", help="Path to baseline .hwpx/.hwp/.pdf")
    compare_parser.add_argument("after", help="Path to edited .hwpx/.hwp/.pdf")
    compare_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for rendered comparison artifacts",
    )
    compare_parser.add_argument(
        "--before-pdf",
        help="Existing baseline PDF exported by Hancom Office",
    )
    compare_parser.add_argument(
        "--after-pdf",
        help="Existing edited PDF exported by Hancom Office",
    )
    compare_parser.add_argument(
        "--export-cmd",
        help=(
            "Export command template with {input} and {output} placeholders. "
            f"If omitted, {EXPORT_ENV_NAME} is used."
        ),
    )
    compare_parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Render DPI for PNG output (default: 180)",
    )
    compare_parser.add_argument(
        "--min-text-similarity",
        type=float,
        default=0.98,
        help="Minimum normalized text similarity for pass (default: 0.98)",
    )
    compare_parser.add_argument(
        "--max-page-diff",
        type=float,
        default=0.35,
        help="Maximum per-page pixel diff ratio for pass (default: 0.35)",
    )
    compare_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when warnings are detected",
    )
    compare_parser.add_argument(
        "--json-output",
        help="Optional JSON summary output path",
    )
    compare_parser.set_defaults(func=command_compare)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.debug)

    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
