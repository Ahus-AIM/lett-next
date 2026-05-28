from __future__ import annotations

import torch
import torch.nn.functional as F

from . import recipe
from .models import _final_logits


class DeepSupervisionLoss(torch.nn.Module):
    def __init__(self, base_loss: torch.nn.Module, weights: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights
        self.latest_component_losses: dict[str, float] = {}

    def forward(self, outputs, targets):
        if not isinstance(outputs, (list, tuple)):
            loss = self.base_loss(outputs, targets)
            self.latest_component_losses = {"main_loss": float(loss.detach().item())}
            return loss
        total = None
        total_weight = 0.0
        latest: dict[str, float] = {}
        for index, output in enumerate(outputs):
            weight = float(self.weights[index] if index < len(self.weights) else self.weights[-1])
            target = targets
            if tuple(output.shape[2:]) != tuple(targets.shape[2:]):
                target = F.interpolate(targets, size=output.shape[2:], mode="nearest")
            component = self.base_loss(output, target)
            latest[f"deep_supervision_loss_{index}"] = float(component.detach().item())
            total = component * weight if total is None else total + component * weight
            total_weight += weight
        if total is None or total_weight <= 0.0:
            raise ValueError("Deep supervision loss received no outputs")
        loss = total / total_weight
        latest["main_loss"] = float(loss.detach().item())
        self.latest_component_losses = latest
        return loss


class DiceBCELoss(torch.nn.Module):
    def __init__(self, dice_weight: float, bce_weight: float, smooth: float = 1e-6) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)
        self.latest_component_losses: dict[str, float] = {}

    def forward(self, logits, targets):
        targets = targets.float()
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.ndim))
        intersection = torch.sum(probabilities * targets, dim=dims)
        denominator = torch.sum(probabilities + targets, dim=dims)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice = dice_loss.mean()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        self.latest_component_losses = {
            "train_dice_loss": float(dice.detach().item()),
            "train_bce_loss": float(bce.detach().item()),
        }
        return self.dice_weight * dice + self.bce_weight * bce


class AnatomyAuxiliaryLoss(torch.nn.Module):
    def __init__(
        self,
        main_loss: torch.nn.Module,
        aux_weight: float,
        ignore_index: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.main_loss = main_loss
        self.aux_weight = float(aux_weight)
        self.ignore_index = int(ignore_index)
        self.num_classes = int(num_classes)
        self.latest_component_losses: dict[str, float] = {}

    def forward(
        self,
        outputs,
        targets: torch.Tensor,
        aux_targets: torch.Tensor,
        aux_valid_mask: torch.Tensor,
        main_loss_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(outputs, dict):
            main_outputs = outputs["main"]
            aux_outputs = outputs.get("aux")
        else:
            main_outputs = outputs
            aux_outputs = None
        main = self.main_loss(main_outputs, targets)
        if main_loss_weight is not None:
            main = main * torch.mean(main_loss_weight.float())
        latest = {"main_loss": float(main.detach().item())}
        if aux_outputs is None:
            self.latest_component_losses = latest
            return main
        aux_logits = _final_logits(aux_outputs)
        if tuple(aux_logits.shape[2:]) != tuple(aux_targets.shape[1:]):
            aux_logits = F.interpolate(aux_logits, size=aux_targets.shape[1:], mode="trilinear", align_corners=False)
        valid_targets = aux_targets.long().clone()
        valid_targets[~aux_valid_mask.bool()] = self.ignore_index
        valid = aux_valid_mask.bool() & (valid_targets != self.ignore_index)
        valid = valid & (valid_targets >= 0) & (valid_targets < self.num_classes)
        if not torch.any(valid):
            latest["train_aux_ce_loss"] = 0.0
            latest["train_aux_dice_loss"] = 0.0
            latest["train_aux_loss"] = 0.0
            self.latest_component_losses = latest
            return main
        ce = F.cross_entropy(aux_logits, valid_targets, ignore_index=self.ignore_index)
        probabilities = torch.softmax(aux_logits, dim=1)
        one_hot = F.one_hot(valid_targets.clamp_min(0), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        one_hot = one_hot.to(device=aux_logits.device, dtype=probabilities.dtype)
        valid_float = valid[:, None, ...].to(device=aux_logits.device, dtype=probabilities.dtype)
        reduce_dims = (2, 3, 4)
        intersection = torch.sum(probabilities * one_hot * valid_float, dim=reduce_dims)
        denominator = torch.sum((probabilities + one_hot) * valid_float, dim=reduce_dims)
        target_mass = torch.sum(one_hot * valid_float, dim=reduce_dims)
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        present = target_mass > 0
        dice = dice_loss[present].mean() if torch.any(present) else torch.zeros((), device=aux_logits.device)
        aux = 0.5 * ce + 0.5 * dice
        latest["train_aux_ce_loss"] = float(ce.detach().item())
        latest["train_aux_dice_loss"] = float(dice.detach().item())
        latest["train_aux_loss"] = float(aux.detach().item())
        self.latest_component_losses = latest
        return main + self.aux_weight * aux


def make_training_loss(*, deep_supervision: bool, aux_class_map) -> torch.nn.Module:
    base_loss = DiceBCELoss(dice_weight=recipe.DICE_WEIGHT, bce_weight=recipe.BCE_WEIGHT)
    main_loss: torch.nn.Module = DeepSupervisionLoss(base_loss) if deep_supervision else base_loss
    if aux_class_map is None:
        return main_loss
    return AnatomyAuxiliaryLoss(
        main_loss,
        aux_weight=recipe.PANTS_AUX_LOSS_WEIGHT,
        ignore_index=aux_class_map.ignore_index,
        num_classes=aux_class_map.num_classes,
    )
