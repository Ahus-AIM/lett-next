# Author notes — not for submission

The response drafts in this folder describe edits already applied to the manuscript on `paper/rebuttal-revision`. The original reviews are preserved as `review-1.md` through `review-3.md`. Table and figure references were checked against the rebuilt PDF.

- Reviewer 1: the upstream local-evaluation correction is verified in the Ahus-AIM/FLARE25 repository. Its README reports revised public-validation DSC 69.7 ± 16.8% and NSD 61.8 ± 19.1%. The response cites these upstream results, which were not independently rerun here. Table 2 now uses the revised validation figures. Its approximate score of 65.8% is derived from the rounded upstream DSC and NSD means, not an exact independently measured score. Hidden-test figures remain organizer-reported. The prompt-removal diagnostic tests inference-time removal, not training without prompts.
- Reviewer 2: the prompt-removal diagnostic answers Reviewer 1's explicit request. No new external-validation or multi-seed claim is made.
- Reviewer 3: the case table is a new inference reproduction, not recovered historical logs. Input and checkpoint hashes match the control. All first-pass masks match exactly. Local NSD uses a 2 mm tolerance. The triggered prompt is label 1 in all three cases. Cases 0057 and 0067 have two targets, so their case-level changes differ from affected-lesion changes (+1.02 and −5.25 pp). Case 0095 has one target.
- Figure 5 retains the original training-set illustrations. These are separate from the three public-validation cases and are not independent validation evidence.

Evidence remains in the original experiment checkout under `results/prompt_channel_diagnostic/full-20261110/` and `results/autozoom_case_analysis/full-20261111/`. Raw predictions and experiment code have not been added to this paper-review worktree.

Before submission, check the venue's rebuttal length and formatting rules. Table and figure numbers may need updating if the manuscript layout changes.
