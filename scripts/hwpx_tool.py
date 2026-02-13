#!/usr/bin/env python3
"""HWPX package utility CLI."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile
import xml.etree.ElementTree as ET

REQUIRED_FILES = (
    "mimetype",
    "version.xml",
    "settings.xml",
    "Contents/content.hpf",
    "Contents/header.xml",
)
SECTION_PATTERN = re.compile(r"^Contents/section(\d+)\.xml$")


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    if debug:
        # TODO: Remove this debug breadcrumb after enough production runs.
        logging.debug("Debug logging enabled")


def local_name(name: str) -> str:
    if "}" in name:
        return name.rsplit("}", 1)[-1]
    if ":" in name:
        return name.rsplit(":", 1)[-1]
    return name


def section_files_from_names(names: Iterable[str]) -> List[str]:
    indexed: List[tuple[int, str]] = []
    for item in names:
        match = SECTION_PATTERN.match(item)
        if not match:
            continue
        indexed.append((int(match.group(1)), item))
    return [name for _, name in sorted(indexed, key=lambda x: x[0])]


def parse_xml_from_zip(archive: ZipFile, path: str) -> ET.Element:
    try:
        with archive.open(path, "r") as handle:
            content = handle.read()
    except KeyError as exc:
        raise FileNotFoundError(f"Missing XML entry: {path}") from exc

    try:
        return ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML in {path}: {exc}") from exc


def find_section_count(header_root: ET.Element) -> Optional[int]:
    for node in header_root.iter():
        if local_name(node.tag) != "head":
            continue
        for key, value in node.attrib.items():
            if local_name(key) != "secCnt":
                continue
            try:
                return int(value)
            except ValueError:
                return None
    return None


def inspect_hwpx(input_path: Path) -> Dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input is not a file: {input_path}")

    try:
        with ZipFile(input_path, "r") as archive:
            names = archive.namelist()
            names_set = set(names)
            required_status = {name: (name in names_set) for name in REQUIRED_FILES}
            section_files = section_files_from_names(names)

            section_count = None
            if "Contents/header.xml" in names_set:
                header_root = parse_xml_from_zip(archive, "Contents/header.xml")
                section_count = find_section_count(header_root)

            missing_required = [k for k, present in required_status.items() if not present]
            warnings: List[str] = []
            if missing_required:
                warnings.append(f"Missing required files: {', '.join(missing_required)}")
            if not section_files:
                warnings.append("No section files found (Contents/section*.xml)")
            if section_count is not None and section_count != len(section_files):
                warnings.append(
                    f"header.xml secCnt={section_count}, section files={len(section_files)}"
                )

            return {
                "input": str(input_path),
                "entries": len(names),
                "required_files": required_status,
                "section_files": section_files,
                "section_count_from_header": section_count,
                "warnings": warnings,
            }
    except BadZipFile as exc:
        raise ValueError(f"Not a valid ZIP/HWPX file: {input_path}") from exc


def collect_paragraph_text(section_root: ET.Element) -> List[str]:
    paragraphs: List[str] = []
    for paragraph in section_root.iter():
        if local_name(paragraph.tag) != "p":
            continue

        fragments: List[str] = []
        for node in paragraph.iter():
            if local_name(node.tag) != "t":
                continue
            text = "".join(node.itertext())
            if text:
                fragments.append(text)
        merged = "".join(fragments).strip()
        if merged:
            paragraphs.append(merged)
    return paragraphs


def extract_text(input_path: Path) -> List[str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input is not a file: {input_path}")

    try:
        with ZipFile(input_path, "r") as archive:
            names = archive.namelist()
            sections = section_files_from_names(names)
            if not sections:
                raise ValueError("No section files found for text extraction")

            extracted: List[str] = []
            for section in sections:
                logging.debug("Parsing section file: %s", section)
                section_root = parse_xml_from_zip(archive, section)
                extracted.extend(collect_paragraph_text(section_root))
            return extracted
    except BadZipFile as exc:
        raise ValueError(f"Not a valid ZIP/HWPX file: {input_path}") from exc


def unpack_hwpx(input_path: Path, output_dir: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input is not a file: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(input_path, "r") as archive:
            archive.extractall(output_dir)
    except BadZipFile as exc:
        raise ValueError(f"Not a valid ZIP/HWPX file: {input_path}") from exc


def pack_hwpx(source_dir: Path, output_path: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source is not a directory: {source_dir}")

    file_list = sorted(path for path in source_dir.rglob("*") if path.is_file())
    if not file_list:
        raise ValueError(f"No files to pack under: {source_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w") as archive:
        mimetype_path = source_dir / "mimetype"
        if mimetype_path.exists() and mimetype_path.is_file():
            # Keep mimetype first and uncompressed for better compatibility.
            archive.write(mimetype_path, arcname="mimetype", compress_type=ZIP_STORED)

        for file_path in file_list:
            relative_path = file_path.relative_to(source_dir).as_posix()
            if relative_path == "mimetype":
                continue
            archive.write(file_path, arcname=relative_path, compress_type=ZIP_DEFLATED)


def command_inspect(args: argparse.Namespace) -> int:
    result = inspect_hwpx(Path(args.input))
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def command_extract_text(args: argparse.Namespace) -> int:
    paragraphs = extract_text(Path(args.input))
    output_text = "\n".join(paragraphs)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)
    return 0


def command_unpack(args: argparse.Namespace) -> int:
    unpack_hwpx(Path(args.input), Path(args.output_dir))
    print(f"Unpacked to: {args.output_dir}")
    return 0


def command_pack(args: argparse.Namespace) -> int:
    pack_hwpx(Path(args.source_dir), Path(args.output))
    print(f"Packed to: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, extract, unpack, and repack HWPX packages."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect HWPX package structure")
    inspect_parser.add_argument("input", help="Path to input .hwpx")
    inspect_parser.add_argument(
        "--json-output",
        help="Optional path to save JSON summary",
    )
    inspect_parser.set_defaults(func=command_inspect)

    text_parser = subparsers.add_parser(
        "extract-text",
        help="Extract paragraph text from section XML files",
    )
    text_parser.add_argument("input", help="Path to input .hwpx")
    text_parser.add_argument("--output", help="Optional path to save extracted text")
    text_parser.set_defaults(func=command_extract_text)

    unpack_parser = subparsers.add_parser("unpack", help="Unpack HWPX package to directory")
    unpack_parser.add_argument("input", help="Path to input .hwpx")
    unpack_parser.add_argument("--output-dir", required=True, help="Directory to extract into")
    unpack_parser.set_defaults(func=command_unpack)

    pack_parser = subparsers.add_parser("pack", help="Pack directory into .hwpx")
    pack_parser.add_argument("source_dir", help="Directory containing unpacked HWPX files")
    pack_parser.add_argument("output", help="Path for output .hwpx")
    pack_parser.set_defaults(func=command_pack)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.debug)

    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
