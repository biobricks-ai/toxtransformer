import math
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning-rate schedule that does NOT depend on optimizer.base_lrs.

    Behavior:
      - Step 0..warmup_steps: linear from min_lr -> max_lr
      - Then: cosine annealing with warm restarts between max_lr and min_lr
              using initial cycle length T_0 and multiplier T_mult.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        cosine_cycle_length: int,
        min_lr: float,
        max_lr: float,
        cosine_t_mult: int = 2,
        last_epoch: int = -1,
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if cosine_cycle_length <= 0:
            raise ValueError("cosine_cycle_length (T_0) must be > 0")
        if not (min_lr >= 0 and max_lr > 0 and max_lr >= min_lr):
            raise ValueError("Require 0 <= min_lr <= max_lr and max_lr > 0")
        if cosine_t_mult < 1:
            raise ValueError("cosine_t_mult must be >= 1")

        self.warmup_steps = int(warmup_steps)
        self.T_0 = int(cosine_cycle_length)
        self.T_mult = int(cosine_t_mult)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)

        # Cosine cycle tracking (relative to the start of cosine phase)
        self._cycle_start = 0           # offset (in cosine-phase steps) where current cycle began
        self._cycle_length = self.T_0   # length of the current cycle in steps

        # Set optimizer LR to min_lr immediately to avoid any dependence on base LR
        for group in optimizer.param_groups:
            group["lr"] = self.min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_linear_warmup(self, step: int) -> float:
        # step ranges 0..warmup_steps
        if self.warmup_steps == 0:
            return self.max_lr
        progress = step / self.warmup_steps
        return self.min_lr + (self.max_lr - self.min_lr) * progress

    def _lr_cosine(self, cos_step: int) -> float:
        """
        cos_step: number of steps since the cosine phase began (>= 0).
        Handles warm restarts with T_mult.
        """
        # Advance cycles until cos_step falls inside the current cycle
        while cos_step - self._cycle_start >= self._cycle_length:
            self._cycle_start += self._cycle_length
            self._cycle_length *= self.T_mult

        # Position within current cycle [0, 1)
        t = (cos_step - self._cycle_start) / self._cycle_length
        # Cosine from max_lr (t=0) to min_lr (t=1)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * t))

    def get_lr(self):
        """
        Return absolute LRs for each param group (same LR for all groups),
        ignoring optimizer.base_lrs entirely.
        """
        # In PyTorch, on first scheduler.step(), last_epoch goes -1 -> 0
        step = self.last_epoch

        if step <= self.warmup_steps:
            lr = self._lr_linear_warmup(step)
        else:
            cos_step = step - self.warmup_steps - 1  # after reaching exactly max at warmup end
            # Ensure the very first post-warmup step starts at max_lr
            if cos_step < 0:
                lr = self.max_lr
            else:
                lr = self._lr_cosine(cos_step)

        return [lr for _ in self.optimizer.param_groups]

    def state_dict(self):
        # include our extra fields so load_state_dict works correctly
        state = super().state_dict()
        state.update({
            "_cycle_start": self._cycle_start,
            "_cycle_length": self._cycle_length,
            "warmup_steps": self.warmup_steps,
            "T_0": self.T_0,
            "T_mult": self.T_mult,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
        })
        return state

    def load_state_dict(self, state_dict):
        self._cycle_start = state_dict.pop("_cycle_start", 0)
        self._cycle_length = state_dict.pop("_cycle_length", self.T_0)
        self.warmup_steps = state_dict.pop("warmup_steps", self.warmup_steps)
        self.T_0 = state_dict.pop("T_0", self.T_0)
        self.T_mult = state_dict.pop("T_mult", self.T_mult)
        self.min_lr = state_dict.pop("min_lr", self.min_lr)
        self.max_lr = state_dict.pop("max_lr", self.max_lr)
        super().load_state_dict(state_dict)


class WarmupCosineThenPlateau(_LRScheduler):
    """
    Learning-rate schedule that does NOT depend on optimizer.base_lrs.

    Phases:
      1) Warmup (steps 0..warmup_steps): linear from min_lr -> max_lr
      2) Cosine (single cycle of length cosine_cycle_length): max_lr -> min_lr
      3) Reduce-on-Plateau: starts at min_lr (from cosine), can decay further
         down to plateau_min_lr using ReduceLROnPlateau semantics.

    Notes:
      - This intentionally uses a SINGLE cosine cycle (no warm restarts).
      - Call .step(metric) during training. The 'metric' is used ONLY in the
        plateau phase; it is ignored earlier.
      - Same LR is applied to all param groups.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        cosine_cycle_length: int,
        min_lr: float,
        max_lr: float,
        # Plateau settings
        plateau_mode: str = "min",          # "min" or "max"
        plateau_factor: float = 0.5,        # multiplicative decay when plateau
        plateau_patience: int = 10,         # epochs/steps without improvement
        plateau_threshold: float = 1e-4,
        plateau_threshold_mode: str = "rel",# "rel" or "abs"
        plateau_cooldown: int = 0,
        plateau_min_lr: float = 0.0,        # final floor for plateau phase
        plateau_eps: float = 1e-8,
        plateau_verbose: bool = False,
        last_epoch: int = -1,
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if cosine_cycle_length <= 0:
            raise ValueError("cosine_cycle_length must be > 0")
        if not (min_lr >= 0 and max_lr > 0 and max_lr >= min_lr):
            raise ValueError("Require 0 <= min_lr <= max_lr and max_lr > 0")
        if plateau_factor <= 0.0 or plateau_factor >= 1.0:
            raise ValueError("plateau_factor must be in (0,1)")

        self.optimizer = optimizer

        # Phase boundaries
        self.warmup_steps = int(warmup_steps)
        self.cosine_len = int(cosine_cycle_length)

        # Cosine endpoints
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)

        # Plateau config
        self.plateau_cfg = dict(
            mode=plateau_mode,
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode=plateau_threshold_mode,
            cooldown=plateau_cooldown,
            min_lr=plateau_min_lr,
            eps=plateau_eps,
        )

        # Phase tracking
        self._phase = "warmup"  # "warmup" | "cosine" | "plateau"
        self._plateau = None    # created lazily at transition
        self._current_lr = self.min_lr

        # Initialize optimizer LR to min_lr to avoid dependence on base_lrs
        for g in self.optimizer.param_groups:
            g["lr"] = self._current_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    # ---------- helpers ----------

    def _set_lr(self, lr: float):
        self._current_lr = float(lr)
        for g in self.optimizer.param_groups:
            g["lr"] = self._current_lr

    def _warmup_lr(self, step: int) -> float:
        if self.warmup_steps == 0:
            return self.max_lr
        progress = step / self.warmup_steps
        return self.min_lr + (self.max_lr - self.min_lr) * progress

    def _cosine_lr(self, k: int) -> float:
        """
        k = number of steps completed within cosine phase (0..cosine_len)
        t in [0,1]: 0 -> max_lr, 1 -> min_lr
        """
        t = min(max(k / self.cosine_len, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * t))

    def _maybe_update_phase(self):
        step = self.last_epoch  # after super().step() increments
        if self._phase == "warmup" and step > self.warmup_steps:
            self._phase = "cosine"

        # Cosine runs for exactly 'cosine_len' steps after warmup.
        # Transition to plateau AFTER finishing the cosine cycle.
        if self._phase == "cosine":
            cos_k = step - (self.warmup_steps + 1)  # 0-based within cosine
            if cos_k >= self.cosine_len:
                # Lock LR to cosine minimum, then switch to plateau
                self._set_lr(self.min_lr)
                self._phase = "plateau"
                # Create ReduceLROnPlateau starting from current LR
                self._plateau = ReduceLROnPlateau(self.optimizer, **self.plateau_cfg)

    # ---------- public API ----------

    def get_lr(self):
        """
        Return absolute LRs for each param group (same LR for all groups).
        PyTorch calls this right after .step() increments last_epoch.
        """
        step = self.last_epoch

        if self._phase == "warmup":
            # step goes -1 -> 0 on first call; warmup defined for 0..warmup_steps
            lr = self._warmup_lr(step)
            self._set_lr(lr)

        elif self._phase == "cosine":
            # Ensure first post-warmup step is exactly max_lr
            k = step - (self.warmup_steps + 1)
            if k < 0:
                lr = self.max_lr
            else:
                lr = self._cosine_lr(k)
            self._set_lr(lr)

        elif self._phase == "plateau":
            # Plateau phase: get_lr should be a no-op (ReduceLROnPlateau has already
            # mutated optimizer.param_groups['lr'] in our .step(metric)).
            # Just mirror current optimizer LR.
            self._current_lr = float(self.optimizer.param_groups[0]["lr"])
            lr = self._current_lr
        else:
            raise RuntimeError(f"Unknown phase {self._phase}")

        return [self._current_lr for _ in self.optimizer.param_groups]

    def step(self, metrics=None):
        """
        Usage:
          - During warmup/cosine: call with no metrics (ignored).
          - During plateau: call with validation metric (min or max according to mode).
        Order: call scheduler.step(val_metric) AFTER optimizer.step() for epoch/step-based usage.
        """
        # If already in plateau, first feed metric to inner scheduler so it can
        # update the optimizer LR; then we advance our epoch and mirror it.
        if self._phase == "plateau" and self._plateau is not None:
            # Clamp via ReduceLROnPlateau's own min_lr setting
            self._plateau.step(metrics)

        # Advance epoch + (re)apply LR from get_lr()
        super().step()
        # Possibly transition phases based on new 'last_epoch'
        self._maybe_update_phase()

    # ---------- state dict ----------

    def state_dict(self):
        state = super().state_dict()
        state.update({
            "_phase": self._phase,
            "_current_lr": self._current_lr,
            "warmup_steps": self.warmup_steps,
            "cosine_len": self.cosine_len,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "plateau_cfg": self.plateau_cfg,
            "_plateau_state": (self._plateau.state_dict() if self._plateau is not None else None),
        })
        return state

    def load_state_dict(self, state_dict):
        self._phase = state_dict.pop("_phase", "warmup")
        self._current_lr = state_dict.pop("_current_lr", self.min_lr)
        self.warmup_steps = state_dict.pop("warmup_steps", self.warmup_steps)
        self.cosine_len = state_dict.pop("cosine_len", self.cosine_len)
        self.min_lr = state_dict.pop("min_lr", self.min_lr)
        self.max_lr = state_dict.pop("max_lr", self.max_lr)
        self.plateau_cfg = state_dict.pop("plateau_cfg", self.plateau_cfg)
        plateau_state = state_dict.pop("_plateau_state", None)

        # Restore base class bits
        super().load_state_dict(state_dict)

        # Recreate/restore inner plateau if needed
        if self._phase == "plateau":
            self._plateau = ReduceLROnPlateau(self.optimizer, **self.plateau_cfg)
            if plateau_state is not None:
                self._plateau.load_state_dict(plateau_state)

        # Re-apply current LR to optimizer
        self._set_lr(self._current_lr)
