from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lett_next.config import ExperimentConfig
from lett_next.data import (
    CaseRecord,
    LEGACY_SPLIT_SOURCE_CACHE_VERSION,
    LEGACY_SPLIT_TARGET_CACHE_VERSION,
    load_manifest,
    load_split_case_cache,
    split_case_source_cache_key,
    split_case_source_cache_path,
    split_case_target_cache_path,
)
from lett_next.inference import _autozoom_next_zoom
from lett_next.metrics import compute_labeled_case_metrics
from lett_next.models import _final_logits, build_model
from lett_next.pipeline import _make_loss
from lett_next.submission_validation import default_validation_inputs, resolve_validation_inputs

import train
import prepare
from lett_next import recipe


def _json_shape(value):
    if isinstance(value, tuple):
        return [_json_shape(item) for item in value]
    if isinstance(value, list):
        return [_json_shape(item) for item in value]
    return value


class ReproducibilityContractTests(unittest.TestCase):
    def test_predict_script_uses_submitted_adaptive_autozoom_defaults(self) -> None:
        text = (REPO_ROOT / "predict.sh").read_text(encoding="utf-8")
        self.assertIn('TASK2_THRESHOLD="${TASK2_THRESHOLD:-0.35}"', text)
        self.assertIn('TASK2_AUTOZOOM_ENABLE="${TASK2_AUTOZOOM_ENABLE:-1}"', text)
        self.assertIn('TASK2_AUTOZOOM_MAX_PASSES="${TASK2_AUTOZOOM_MAX_PASSES:-2}"', text)
        self.assertIn('TASK2_AUTOZOOM_MIN_PASSES="${TASK2_AUTOZOOM_MIN_PASSES:-1}"', text)
        self.assertIn('TASK2_AUTOZOOM_GROWTH_ZYX="${TASK2_AUTOZOOM_GROWTH_ZYX:-1.25,1.15,1.15}"', text)
        self.assertIn('TASK2_AUTOZOOM_SCALE_MODE="${TASK2_AUTOZOOM_SCALE_MODE:-adaptive}"', text)
        self.assertIn('TASK2_AUTOZOOM_REFINE_DISABLE="${TASK2_AUTOZOOM_REFINE_DISABLE:-1}"', text)
        self.assertIn('--model-name "${TASK2_MODEL_NAME:-mednextv2_f32}"', text)

    def test_python_submission_defaults_match_docker_defaults(self) -> None:
        self.assertEqual(recipe.SUBMISSION_THRESHOLD, 0.35)
        self.assertEqual(recipe.SUBMISSION_AUTOZOOM_KWARGS["autozoom_enable"], True)
        self.assertEqual(recipe.SUBMISSION_AUTOZOOM_KWARGS["autozoom_max_passes"], 2)
        self.assertEqual(recipe.SUBMISSION_AUTOZOOM_KWARGS["autozoom_min_passes"], 1)
        self.assertEqual(recipe.SUBMISSION_AUTOZOOM_KWARGS["autozoom_growth_zyx"], (1.25, 1.15, 1.15))
        self.assertEqual(recipe.SUBMISSION_AUTOZOOM_KWARGS["autozoom_scale_mode"], "adaptive")
        self.assertEqual(recipe.SUBMISSION_AUTOZOOM_KWARGS["autozoom_refine_enable"], False)

    def test_submission_validation_uses_public_validation_npz_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            validation_root = root / "data/validation_npz"
            public_npz = validation_root / "validation_public_npz"
            public_npz.mkdir(parents=True)
            self.assertEqual(default_validation_inputs(root), public_npz)
            self.assertEqual(resolve_validation_inputs(root, validation_root), public_npz)

    def test_submitted_checkpoint_is_inference_only(self) -> None:
        submitted = REPO_ROOT / "artifacts/checkpoint.pt"
        state = torch.load(submitted, map_location="cpu", weights_only=False)
        self.assertEqual(sorted(state), ["best_score", "config", "config_snapshot", "model_name", "model_state"])
        config = state["config"]
        self.assertEqual(config["model_name"], "mednextv2_f32")
        self.assertEqual(config["crop_size_zyx"], (72, 160, 160))
        metadata_text = repr({key: value for key, value in state.items() if key != "model_state"})
        for private_marker in ("/home/", "wandb", "optimizer_state", "history", "run_id"):
            self.assertNotIn(private_marker, metadata_text)

    def test_config_matches_canonical_stable_fields(self) -> None:
        config = ExperimentConfig.from_source(REPO_ROOT, REPO_ROOT / "configs/lett_next.yaml")
        expected = {
            "model_name": "mednextv2_f32",
            "deep_supervision": True,
            "epochs": 25,
            "eval_every": 3,
            "batch_size": 6,
            "learning_rate": 0.001,
            "num_workers": 8,
            "crop_size_zyx": [72, 160, 160],
            "target_spacing_xyz": [1.0, 1.0, 2.4],
            "prompt_sigma": 2.0,
            "threshold": 0.5,
            "seed": 43,
            "anatomy_aux_enable": True,
            "anatomy_aux_loss_weight": 0.25,
            "train_crop_mode": "mixed_recist_prompt",
            "mixed_prompt_base_fraction": 0.6,
            "mixed_prompt_multifov_fraction": 0.3,
            "mixed_recist_bbox_fit_tail_fraction": 0.1,
            "mixed_recist_multifov_scales_zyx": [
                [1.0, 1.0, 1.0],
                [1.15, 1.4, 1.4],
                [1.35, 1.8, 1.8],
                [1.35, 2.4, 2.4],
            ],
            "ensure_recist_endpoints_inside_crop": True,
            "recist_endpoint_margin_mm": 15.0,
        }
        actual = config.to_dict()
        for key, expected_value in expected.items():
            self.assertEqual(_json_shape(actual[key]), expected_value, key)
        self.assertEqual(recipe.POSTPROCESS_KWARGS["postprocess_mode"], "prompt_component")
        self.assertEqual(recipe.VALIDATION_AUTOZOOM_KWARGS["autozoom_enable"], True)
        self.assertEqual(recipe.VALIDATION_AUTOZOOM_KWARGS["autozoom_max_passes"], 3)
        self.assertEqual(config.train_nproc_per_node, 3)
        self.assertEqual(config.train_cuda_visible_devices, "0,1,2")

    def test_bundled_manifests_are_portable_and_resolve_under_package_root(self) -> None:
        manifest_path = (
            REPO_ROOT
            / "data/prepared/train_manifest.json"
        )
        text = manifest_path.read_text(encoding="utf-8")
        payload = json.loads(text)
        first = payload["records"][0]
        self.assertEqual(first["image_path"], "data/train_npz/CT_Lesion_000001_02_01_008-023.npz")
        self.assertNotIn("/home/sebastian", text)
        self.assertNotIn("/files3/dataset/PanTS", text)
        records = load_manifest(manifest_path)
        self.assertTrue(records[0].image_path.startswith(str(REPO_ROOT / "data/train_npz")))

    def test_train_launcher_uses_configured_three_gpu_shape(self) -> None:
        config = ExperimentConfig.from_source(REPO_ROOT, REPO_ROOT / "configs/lett_next.yaml")
        command = train._launcher_command(config)
        self.assertIn("--nproc_per_node=3", command)
        self.assertEqual(config.train_run_suffix, "lett-next-mednext-f32-xy1p0-z2p4-z72xy160-25ep-b6-w8-3gpu")

    def test_entrypoints_accept_external_config_override(self) -> None:
        override = Path("/tmp/lett-next-override.yaml")
        self.assertEqual(train._resolve_config_path(override), override)
        self.assertEqual(prepare._resolve_config_path(override), override)

    def test_split_cache_falls_back_by_case_id_when_path_key_changes(self) -> None:
        spacing = (1.0, 1.0, 2.4)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            raw_path = root / "raw_exists.npz"
            raw_path.write_bytes(b"raw placeholder")
            old_record = CaseRecord(
                case_id="CASE_001",
                image_path=str(root / "old_manifest_path.npz"),
                label_path=None,
                spacing_xyz=[1.0, 1.0, 2.4],
                recist_endpoints_xyz=[[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
                recist_slice_index=1,
                split="train",
                source="unit",
                prompt_source="recist",
                target_label_value=1,
                target_component_id=7,
            )
            current_record = CaseRecord(
                case_id=old_record.case_id,
                image_path=str(raw_path),
                label_path=None,
                spacing_xyz=old_record.spacing_xyz,
                recist_endpoints_xyz=old_record.recist_endpoints_xyz,
                recist_slice_index=old_record.recist_slice_index,
                split=old_record.split,
                source=old_record.source,
                prompt_source=old_record.prompt_source,
                target_label_value=old_record.target_label_value,
                target_component_id=old_record.target_component_id,
            )
            source_key = split_case_source_cache_key(old_record, target_spacing_xyz=spacing)
            source_path = split_case_source_cache_path(old_record, cache_dir=cache_dir, target_spacing_xyz=spacing)
            target_path = split_case_target_cache_path(old_record, cache_dir=cache_dir, target_spacing_xyz=spacing)
            source_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            mask = np.zeros((2, 2, 2), dtype=np.uint8)
            mask[1, 1, 1] = 1
            np.savez_compressed(
                source_path,
                cache_version=np.asarray(LEGACY_SPLIT_SOURCE_CACHE_VERSION, dtype=np.str_),
                source_cache_key=np.asarray(source_key, dtype=np.str_),
                image=image,
                spacing_xyz=np.asarray(spacing, dtype=np.float32),
            )
            np.savez_compressed(
                target_path,
                cache_version=np.asarray(LEGACY_SPLIT_TARGET_CACHE_VERSION, dtype=np.str_),
                source_cache_key=np.asarray(source_key, dtype=np.str_),
                task2_target_mask=mask,
                recist_endpoints_xyz=np.asarray(old_record.recist_endpoints_xyz, dtype=np.float32),
                recist_slice_index=np.asarray(old_record.recist_slice_index, dtype=np.int32),
                selected_label_value=np.asarray(old_record.target_label_value, dtype=np.int32),
                selected_component_id=np.asarray(old_record.target_component_id, dtype=np.int32),
                selected_voxel_count=np.asarray(int(mask.sum()), dtype=np.int64),
            )

            loaded = load_split_case_cache(
                current_record,
                cache_dir=cache_dir,
                target_spacing_xyz=spacing,
            )

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.recist_target_selection["used_fallback"])
        np.testing.assert_array_equal(loaded.image, image)
        np.testing.assert_array_equal(loaded.recist_target_mask, mask)

    def test_model_forward_and_loss_smoke(self) -> None:
        torch.set_num_threads(1)
        model = build_model("mednextv2_f32", in_channels=2, out_channels=1, deep_supervision=False)
        model.eval()
        inputs = torch.zeros((1, 2, 32, 32, 32), dtype=torch.float32)
        with torch.no_grad():
            logits = _final_logits(model(inputs))
        self.assertEqual(tuple(logits.shape), (1, 1, 32, 32, 32))

        config = ExperimentConfig.from_source(REPO_ROOT, REPO_ROOT / "configs/lett_next.yaml")
        config.anatomy_aux_enable = False
        criterion = _make_loss(config, "cpu")
        loss = criterion(logits, torch.zeros_like(logits))
        self.assertTrue(torch.isfinite(loss))

    def test_labeled_metrics_average_recist_ids(self) -> None:
        gt = np.zeros((4, 8, 8), dtype=np.uint8)
        gt[1, 1:3, 1:3] = 1
        gt[2, 5:7, 5:7] = 2
        pred = np.zeros_like(gt)
        pred[1, 1:3, 1:3] = 1

        metrics = compute_labeled_case_metrics(gt, pred, np.asarray([1.0, 1.0, 1.0]), [1, 2])

        self.assertEqual(metrics["lesion_count"], 2)
        self.assertAlmostEqual(float(metrics["dsc"]), 0.5)
        self.assertAlmostEqual(float(metrics["nsd"]), 0.5)

    def test_adaptive_autozoom_only_scales_to_needed_zoom_for_weak_border_contact(self) -> None:
        zoom = _autozoom_next_zoom(
            zoom_zyx=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            trigger_axes=[0, 1, 2],
            growth_zyx=(1.25, 1.15, 1.15),
            max_zoom_zyx=(4.0, 2.0, 2.0),
            candidate_zoom_zyx=np.asarray([1.05, 1.03, 1.03], dtype=np.float32),
            border_fractions_zyx=[0.001, 0.001, 0.001],
            scale_mode="adaptive",
        )

        np.testing.assert_allclose(zoom, np.asarray([1.05, 1.03, 1.03], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
