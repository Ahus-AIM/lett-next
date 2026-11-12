# Response to Reviewer 3

Thank you for suggesting a closer analysis of the triggered cases. AutoZoom was rerun on all 49 validation cases (84 lesions) using the submitted settings. Expansion triggered in three cases, with the following case-level scores. Every first-pass prediction matched the saved fixed-crop control exactly.

| Case | Fixed crop (%) | AutoZoom (%) | Change (pp) |
|---|---:|---:|---:|
| 0057 | 64.89 | 65.40 | +0.51 |
| 0067 | 82.77 | 80.15 | −2.62 |
| 0095 | 69.97 | 66.32 | −3.64 |

In case 0057, expansion recovered more lesion voxels but also added false positives, giving a small net improvement. In case 0067, both missed lesion voxels and false positives increased. In case 0095, the reduction in false positives was accompanied by more missed lesion voxels and a lower score.

The two losses outweighed the improvement, reducing the overall mean from 75.99% to 75.87% (−0.12 percentage points, 95% paired bootstrap CI: −0.32 to +0.02). This does not establish an average accuracy benefit. Section 5.3 now summarises these mixed outcomes, and Appendix D provides the complete three-case comparison (Table 11) alongside the qualitative illustrations.
