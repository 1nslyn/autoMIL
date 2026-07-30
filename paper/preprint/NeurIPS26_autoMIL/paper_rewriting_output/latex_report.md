# LaTeX Report

- **Compile status:** PASS
- **Manuscript status:** DRAFT — 9 intentional placeholders remain open in
  `placeholder_register.md`
- **Template:** bundled `neurips_2026.sty` with the `preprint` option
- **Source:** `final_paper/main.tex`
- **Bibliography:** `final_paper/references.bib`
- **Compiled PDF:** `final_paper/main.pdf`
- **Delivery alias:** `final_paper/paper.pdf`
- **Compile command:** `/Library/TeX/texbin/latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- **PDF size:** 18 pages total; the main text ends on page 8, before the reference list completes
- **Citation/label guard:** 0 errors, 0 warnings
- **PDF alias check:** `main.pdf` and `paper.pdf` have identical SHA-256 digests
- **Visual check:** pages 1–10 were rasterized and inspected as a contact sheet; the two TikZ diagrams, four main tables, equations, references, and page breaks render without clipping
- **Compile-log note:** no undefined references/citations or overfull boxes remain. LaTeX emits one harmless package-path warning because the final source loads the bundled style from its parent directory.
- **Word output:** intentionally omitted because `paper_spine_config.json` explicitly sets `word_output` to `none`; the requested deliverable is the locally viewable PDF.

The visible placeholder boxes compile cleanly and preserve the exact insertion
points for author metadata, protocol decisions, final results, and release
details. A clean compile is not a final-audit declaration while they remain.
