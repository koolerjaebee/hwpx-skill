---
name: hwpx
description: "Handle HWPX documents (.hwpx) with quality-first workflows: inspect package structure, edit XML safely, render for visual verification, and compare before/after outputs. Use when tasks require high-confidence HWPX editing where layout and rendering fidelity matter more than speed."
---

# HWPX Skill

## Overview

Inspect and modify `.hwpx` as ZIP+XML, then validate quality through render-based checks.  
Prefer slower, deterministic verification before final delivery.

## Quick Start

1. Inspect structure first:
```bash
python3 scripts/hwpx_tool.py inspect /path/to/file.hwpx
```
2. Export baseline PDF with Hancom Office, then render:
```bash
python3 scripts/render_hwpx.py render /path/to/file.hwpx \
  --pdf-path /path/to/file_from_hancom.pdf \
  --output-dir /tmp/hwpx_baseline
```
3. Unpack, edit, and repack:
```bash
python3 scripts/hwpx_tool.py unpack /path/to/file.hwpx --output-dir /tmp/hwpx_work
# edit XML files in /tmp/hwpx_work
python3 scripts/hwpx_tool.py pack /tmp/hwpx_work /path/to/updated.hwpx
```
4. Export edited PDF with Hancom Office, then compare:
```bash
python3 scripts/render_hwpx.py compare \
  /path/to/file.hwpx \
  /path/to/updated.hwpx \
  --before-pdf /path/to/file_from_hancom.pdf \
  --after-pdf /path/to/updated_from_hancom.pdf \
  --output-dir /tmp/hwpx_compare \
  --strict
```

## Hancom PDF Export

Use one of these modes:

1. Manual export (recommended default)
   - Export PDF in Hancom Office and pass `--pdf-path` / `--before-pdf` / `--after-pdf`.
2. Command template export
   - If your environment has a Hancom export CLI, pass `--export-cmd` with placeholders:
   ```bash
   --export-cmd 'your_hancom_export_command --input "{input}" --output "{output}"'
   ```
   - You can also set environment variable `HANCOM_PDF_EXPORT_CMD`.

## Workflow

1. Run `inspect` and stop immediately on broken package structure.
2. Generate baseline PDF with Hancom Office (or configured export command).
3. Run `render` and keep artifacts as baseline evidence.
4. Edit only required XML nodes, then repack.
5. Generate edited PDF with Hancom Office.
6. Run `compare` with `--strict` to gate regressions.
7. Manually review generated PNG pages for alignment, clipping, overlap, and missing objects.

## Commands

- `inspect <input.hwpx> [--json-output PATH] [--debug]`
  - Validate package-level structure and summarize sections.
- `unpack <input.hwpx> --output-dir DIR [--debug]`
  - Extract all package entries for direct XML editing.
- `pack <source_dir> <output.hwpx> [--debug]`
  - Rebuild an HWPX package from an unpacked directory.
- `render <input.hwpx|.pdf> --output-dir DIR [--pdf-path PDF] [--export-cmd TEMPLATE] [--dpi N] [--json-output PATH] [--debug]`
  - Render document as PNG pages using Hancom-generated PDF.
- `compare <before.hwpx|.pdf> <after.hwpx|.pdf> --output-dir DIR [--before-pdf PDF] [--after-pdf PDF] [--export-cmd TEMPLATE] [--strict] [--min-text-similarity F] [--max-page-diff F] [--json-output PATH] [--debug]`
  - Compare before/after via metadata, normalized text similarity, and page-level pixel diff ratio.

## Editing Guardrails

- Never skip render-based verification for production edits.
- Keep path casing and folder structure unchanged when repacking.
- Preserve `Contents/header.xml` consistency with available `section*.xml` files.
- Avoid large refactors of XML formatting unless required by the task.
- Prefer scripted updates over manual ZIP operations for repeatability.
- Keep baseline and edited render artifacts until user approval.

## Reference

Read `references/hwpx-format.md` when you need concrete structure details, required paths, or text node parsing notes.

## Dependencies

Python:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install pillow
```

System tools:
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install -y poppler-utils
```
