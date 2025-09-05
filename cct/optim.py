import math
import torch
from torch import optim


def build_optimizer(model, lr: float, weight_decay: float = 0.1):
    """AdamW optimizer with decoupled weight decay."""
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


class WarmHoldCosineLR:
    """
    Cosine learning rate schedule with warmup, hold, and cosine decay.
    """

    def __init__(self, optimizer, total_steps: int, warmup_frac: float, hold_frac: float, min_lr_ratio: float):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_frac)
        self.hold_steps = int(total_steps * hold_frac)
        self.decay_steps = total_steps - self.warmup_steps - self.hold_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = self.get_lr(base_lr)

    def get_lr(self, base_lr: float):
        if self.step_num < self.warmup_steps:
            return base_lr * (self.step_num / self.warmup_steps)
        elif self.step_num < self.warmup_steps + self.hold_steps:
            return base_lr
        else:
            progress = (self.step_num - self.warmup_steps - self.hold_steps) / max(1, self.decay_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
        
    def state_dict(self):
        return {
            "step_num": self.step_num,
            "base_lrs": self.base_lrs,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "hold_steps": self.hold_steps,
            "decay_steps": self.decay_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, state):
        # restore schedule shape if present
        self.total_steps  = state.get("total_steps",  self.total_steps)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.hold_steps   = state.get("hold_steps",   self.hold_steps)
        self.decay_steps  = state.get("decay_steps",  self.decay_steps)
        self.min_lr_ratio = state.get("min_lr_ratio", self.min_lr_ratio)

        self.step_num = state.get("step_num", 0)
        self.base_lrs = state.get("base_lrs", self.base_lrs)

        # sync LR to the restored step
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = self.get_lr(base_lr)