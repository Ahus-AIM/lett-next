Review:
Paper has a clear and concise writing style with good illustrations.

Strengths of LETT-NeXt

A solid breakdown of performance gains through the controlled ablation studies performed in the paper. The authors show that backbone choice is an import factor when it comes to this challenge.
Paper is very open and transparent about the effectiveness of certain additions (for example, the mentioning of the auxilliary head being a weak training-time regularizer rather than a definitive improvement) rather than inflating their significance.
Test set gains over Lite ENSAM are substantial.
Weaknesses of LETT-NeXt

Conditioning on the RECIST marker seems currently unfounded. What would happen to performance if you don't condition on the RECIST markers, i.e. leaving the prompt channels empty while keeping the center crop? The RECIST markers were a central part of the challenge of course, but I'd be interested to see if they would add anything of value?
In the discussion you mention:
"LETT-NeXt demonstrates that sparse RECIST markers can support efficient 3D lesion segmentation under CPU inference constraints. The model generally localized the marked lesions, while the remaining errors mainly involved lesion extent and boundary placement. This pattern is consistent with the lower NSD than DSC on the public validation set and with the under-segmentation examples in Figure 3. In practice, this suggests that the RECIST marker provides strong localization information, but does not fully resolve the 3D lesion boundary".

If this is true, then why doesn't it hold for Lite ENSAM? Doesn't it receive the same RECIST markers, so wouldn't we expect that the NSD would also be lower for this model?

Rating: 7: Good paper, accept
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct