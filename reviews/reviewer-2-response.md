# Response to Reviewer 2

Thank you for the positive assessment and helpful comments.

**Contribution and prior work.** The introduction now states that LETT-NeXt combines established components for RECIST-guided segmentation under CPU inference constraints. The existing citations attribute prompt-channel conditioning and AutoZoom to nnInteractive and the backbone to MedNeXt-v2. The backbone comparison provides the clearest empirical support for the design choices.

**Boundary accuracy.** Sections 4.1 and 6 now ground the under-segmentation discussion in the qualitative examples in Figure 3. The numerical ordering of DSC and NSD alone does not establish under-segmentation.

**Validation limitations.** Section 6 explicitly acknowledges that reusing the small validation set for model development may produce optimistic estimates. The existing limitations include the absence of external validation and limited hidden-test detail. Hidden-test results assess the submitted system but do not establish external generalization or isolate component effects.

**Marginal gains.** Auxiliary supervision improved the mean score by 0.96 percentage points (95% CI: −0.18 to +2.05), while AutoZoom changed it by −0.12 points (95% CI: −0.32 to +0.02). Neither establishes an aggregate benefit. Section 6 retains the cautious interpretation of auxiliary supervision and now describes AutoZoom as having near-neutral aggregate performance with mixed case-level outcomes. Appendix D provides the detailed AutoZoom comparison.
