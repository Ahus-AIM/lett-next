from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = REPO_ROOT / "PanCancerSeg_Test"
DEFAULT_OUTPUTS = REPO_ROOT / "lett_next_outputs"
CHECKPOINT = REPO_ROOT / "artifacts/checkpoint.pt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from lett_next import recipe
    from lett_next.submission_cpu import run_submission_prediction

    summary = run_submission_prediction(
        inputs_dir=DEFAULT_INPUTS,
        outputs_dir=DEFAULT_OUTPUTS,
        checkpoint_path=CHECKPOINT,
        model_name="mednextv2_f32",
        crop_size_zyx=(128, 160, 160),
        prompt_sigma=2.0,
        threshold=recipe.SUBMISSION_THRESHOLD,
        deep_supervision=False,
        allow_untrained_smoke=False,
        output_format="nii",
        threads=12,
        time_limit_seconds=60.0,
        **recipe.SUBMISSION_AUTOZOOM_KWARGS,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
