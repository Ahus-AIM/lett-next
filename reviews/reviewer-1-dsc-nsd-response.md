# Response to Reviewer 1

Thank you for the constructive feedback.

**Value of the prompt channels.** Section 5.4 now reports the requested test on all 49 validation cases (84 lesions) using the frozen submitted checkpoint. Both prompt channels were zeroed while retaining identical RECIST-centered CT crops, threshold 0.35, and prompt-aware postprocessing. AutoZoom was disabled in both arms.

| Network prompt channels | DSC (%) | NSD (%) | Mean score (%) |
|---|---:|---:|---:|
| Original | 79.47 | 72.51 | 75.99 |
| Both zeroed | 31.52 | 24.64 | 28.08 |

The score dropped by 47.91 percentage points (95% paired bootstrap CI: 39.96–55.63). This supports reliance on the explicit channels beyond cropping and postprocessing. The test removes prompts only at inference and does not show how a model trained without them would perform. These results appear in Table 7. Evaluation details are in Section 3.1, with bootstrap details in Appendix C.

**Interpretation of DSC and NSD.** Thank you for highlighting this discrepancy. The [Lite ENSAM repository](https://github.com/Ahus-AIM/FLARE25) has since added a corrected local evaluator using per-lesion masks and the original physical voxel spacing. The revised public-validation results report NSD of 61.8%, compared with the previously reported 78.9%, and DSC of 69.7%. Table 2 now includes the revised validation results, which no longer show NSD above DSC. Nevertheless, the numerical ordering of these metrics alone does not establish under-segmentation. Sections 4.1 and 6 now ground the boundary-error discussion in the qualitative examples in Figure 3.
