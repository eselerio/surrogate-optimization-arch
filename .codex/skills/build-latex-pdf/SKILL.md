---
name: build-latex-pdf
description: Build PDFs from LaTeX source strictly through terminal or command-line workflows. Use when Codex needs to compile one or more .tex files, regenerate manuscript or supplementary PDFs, validate a LaTeX project by producing a PDF, or archive LaTeX build outputs. This skill stages the source project into a fresh timestamped folder under docs/latex_pdfs, compiles there, and leaves the copied .tex sources together with the generated PDFs in that folder.
---

# Build LaTeX PDF

## Overview

Use the bundled PowerShell script to create a fresh timestamped build folder under `docs\latex_pdfs`, stage a copy of the LaTeX project there, and compile the staged `.tex` files by terminal only.

## Workflow

1. Prefer passing `.tex` files from one LaTeX project folder per invocation.
2. Run the bundled script instead of ad hoc GUI builds or editor buttons.
3. Read the script output and report the timestamped folder plus the generated PDF paths.
4. If the build fails, inspect the log inside the timestamped folder, fix the source, and rerun so the next attempt gets its own fresh archive folder.

## Command

Run the bundled script from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File ".codex\skills\build-latex-pdf\scripts\build_latex_pdf.ps1" -TexFiles "artifacts\wip\manuscript.tex"
```

Build multiple `.tex` files from the same project folder in one archived run:

```powershell
powershell -ExecutionPolicy Bypass -File ".codex\skills\build-latex-pdf\scripts\build_latex_pdf.ps1" `
  -TexFiles "artifacts\wip\manuscript.tex","artifacts\wip\supplementary_material.tex"
```

Use an explicit source root when the `.tex` files live below a larger project tree:

```powershell
powershell -ExecutionPolicy Bypass -File ".codex\skills\build-latex-pdf\scripts\build_latex_pdf.ps1" `
  -SourceRoot "artifacts\wip" `
  -TexFiles "artifacts\wip\manuscript.tex","artifacts\wip\supplementary_material.tex"
```

## Script Behavior

- Always use the terminal or command line only.
- Always create a new timestamped folder in `docs\latex_pdfs` using `yyyyMMdd_HHmmss`.
- Always stage a copy of the source project into that timestamped folder before compiling.
- Always leave the copied `.tex` sources and generated PDFs inside that timestamped folder.
- Copy the project recursively while skipping common generated build artifacts so the staged folder stays focused on source files and outputs.
- Prefer a repo-local `.tools\tectonic\tectonic.exe`, then `tectonic`, then `latexmk`, then `xelatex`, `lualatex`, and `pdflatex`.
- Bootstrap a repo-local standalone `tectonic` automatically on Windows if no CLI LaTeX engine is already available.
- Remove common auxiliary files after a successful build unless `-KeepAuxFiles` is passed.

## Parameters

- `-TexFiles`: Required. One or more `.tex` files to compile.
- `-SourceRoot`: Optional. The project folder to stage into the timestamped build folder. If omitted, the script requires all `.tex` files to come from the same directory and uses that directory.
- `-ProjectRoot`: Optional. Repository root used to resolve `docs\latex_pdfs` and repo-local tooling. If omitted, the script tries `git rev-parse --show-toplevel`, then walks upward for `.git`, then falls back to the current directory.
- `-OutputRoot`: Optional. Override the default `docs\latex_pdfs` destination.
- `-KeepAuxFiles`: Optional. Keep `.aux`, `.toc`, `.out`, `.xdv`, and related build intermediates in the timestamped folder.

## Reporting

After running the script:

- Report the timestamped output folder path.
- Report the generated PDF paths.
- Mention which engine was used if that matters for troubleshooting.
- If the build failed, cite the log path in the timestamped folder.
