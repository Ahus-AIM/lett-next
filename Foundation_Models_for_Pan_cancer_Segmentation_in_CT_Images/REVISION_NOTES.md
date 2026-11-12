# Rebuttal revision

Review `LETT-NeXt-revised.pdf` (20 pages).

- Table 1: corrected optimizer to Adam; Section 3.1 specifies local evaluation conventions.
- Section 5.3: retained a brief summary, referring to the appendix for all three triggered AutoZoom cases.
- Appendix D: combined the three-case public-validation table with the original qualitative figure at reduced size. Its caption now identifies the training-set examples and distinguishes them from the public-validation analysis.
- Section 5.4: added the matched test-time prompt-removal diagnostic (Table 7).
- Introduction, results and discussion: clarified attribution, metric interpretation, validation reuse and uncertain component effects.
- Bounded table widths and resized Figure 3; no overfull boxes, oversized floats or unresolved references remain.

Review Section 5.4 for prompt removal and Appendix D for the compact AutoZoom case analysis. Table 2 uses the revised Lite ENSAM public-validation figures from its corrected evaluator. Its score is marked approximate because the source reports rounded metric means. Hidden-test results remain unchanged. No new experiments were run for this manuscript edit.

Build from this folder: `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex`.
