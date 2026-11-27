"""Run the stationary and continual learning benchmarks for Imp Optimiser."""

import argparse
import json
import math
import os
import random
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

import matplotlib.pyplot as plt

# Reproducibility and numeric utilities

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mean_std(arr: np.ndarray, axis=0) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    return arr.mean(axis=axis), arr.std(axis=axis, ddof=0)

def safe_float(x: Any, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


# φ (anchored sigmoid)

def _sigmoid(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-u))

def phi_np(dloss: float, Apos: float, mu: float, s: float) -> float:
    """
    Anchored sigmoid:
        phi(x) = Apos * (sigmoid((x - mu)/s) - sigmoid((0 - mu)/s))
    """
    s = max(float(s), 1e-12)
    base = float(_sigmoid(np.array([(-mu) / s]))[0])
    out = float(Apos * (float(_sigmoid(np.array([(dloss - mu) / s]))[0]) - base))
    return out

def phi_curve(xs: np.ndarray, phi_params: Tuple[float, float, float]) -> np.ndarray:
    Apos, mu, s = (float(phi_params[0]), float(phi_params[1]), float(phi_params[2]))
    xs = np.asarray(xs, dtype=float)
    s = max(s, 1e-12)
    base = _sigmoid((-mu) / s)
    return Apos * (_sigmoid((xs - mu) / s) - base)

def phi_linear(xs: np.ndarray, slope: float = 0.1) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    return slope * xs

def phi_flat(xs: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    return np.zeros_like(xs)


# Imp optimiser (with history logging)

class ImpOptimiser(Optimizer):
    def __init__(
        self,
        params,
        *,
        alpha_max: float,
        alpha_min: float,
        k: float,
        phi_params: Tuple[float, float, float],
        z0: float = 0.0,
        z_min: float = 0.0,
        z_max: float = 10.0,
        L_est: float = 10.0,
        eps: float = 1e-12,
        dloss_ema_beta: float = 0.9,
        phi_mode: str = "normal",               # normal | disabled | shuffled
        shuffled_dloss_buffer: Optional[List[float]] = None,
        record_history: bool = True,
    ):
        defaults = dict(alpha=float(alpha_max))
        super().__init__(params, defaults)

        self.alpha_max = float(alpha_max)
        self.alpha_min = float(alpha_min)
        self.k = float(k)

        self.z = float(z0)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

        self.L_est = float(L_est)
        self.eps = float(eps)

        self.prev_loss = None
        self.phi_params = tuple(float(x) for x in phi_params)

        self.dloss_ema_beta = float(dloss_ema_beta)
        self._dloss_ema = 0.0

        self.phi_mode = str(phi_mode)
        self._shuffled = list(shuffled_dloss_buffer) if shuffled_dloss_buffer is not None else []
        self._shuf_idx = 0

        self._alpha_sum = 0.0
        self._alpha_min_seen = float("inf")
        self._alpha_max_seen = 0.0
        self._alpha_n = 0

        # Record the optimiser trajectory for the dynamics plot.
        self.record_history = bool(record_history)
        self.history: List[Dict[str, float]] = []

        self._update_alpha_from_z()

    def reset_alpha_stats(self):
        self._alpha_sum = 0.0
        self._alpha_min_seen = float("inf")
        self._alpha_max_seen = 0.0
        self._alpha_n = 0

    def alpha_stats(self) -> Dict[str, float]:
        if self._alpha_n <= 0:
            return {"alpha_mean": float("nan"), "alpha_min": float("nan"), "alpha_max": float("nan")}
        return {
            "alpha_mean": float(self._alpha_sum / self._alpha_n),
            "alpha_min": float(self._alpha_min_seen),
            "alpha_max": float(self._alpha_max_seen),
        }

    def _update_alpha_from_z(self):
        # Exponential “stiffness”: larger z => smaller alpha
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * math.exp(-self.z)
        # Optional theoretical cap (2/L)
        safe_cap = (2.0 / self.L_est) - self.eps
        if safe_cap > 0:
            alpha = min(alpha, safe_cap)
        for group in self.param_groups:
            group["alpha"] = float(alpha)

    def _next_shuffled_dloss(self) -> float:
        if not self._shuffled:
            return 0.0
        v = float(self._shuffled[self._shuf_idx % len(self._shuffled)])
        self._shuf_idx += 1
        return v

    @torch.no_grad()
    def step(self, *, loss: torch.Tensor):
        loss_val = float(loss.detach().item())
        if self.prev_loss is None:
            dloss = 0.0
        else:
            dloss = loss_val - self.prev_loss

        b = self.dloss_ema_beta
        if b > 0.0:
            self._dloss_ema = b * self._dloss_ema + (1.0 - b) * dloss
            dloss_used = self._dloss_ema
        else:
            dloss_used = dloss

        if self.phi_mode == "disabled":
            dz = 0.0
        elif self.phi_mode == "shuffled":
            dz = phi_np(self._next_shuffled_dloss(), *self.phi_params)
        else:
            dz = phi_np(dloss_used, *self.phi_params)

        self.z = float(np.clip(self.z + self.k * dz, self.z_min, self.z_max))
        self._update_alpha_from_z()

        for group in self.param_groups:
            alpha = float(group["alpha"])

            self._alpha_sum += alpha
            self._alpha_min_seen = min(self._alpha_min_seen, alpha)
            self._alpha_max_seen = max(self._alpha_max_seen, alpha)
            self._alpha_n += 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-alpha)

        self.prev_loss = loss_val

        if self.record_history:
            alpha0 = float(self.param_groups[0]["alpha"])
            self.history.append({
                "loss": float(loss_val),
                "dloss": float(dloss_used),
                "z": float(self.z),
                "alpha": float(alpha0),
            })

        return loss


# Model and synthetic data streams

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def stream_task_A(steps: int, batch: int, seed: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    g = torch.Generator()
    g.manual_seed(seed)
    xs, ys = [], []
    for _ in range(steps):
        x = (4 * math.pi) * torch.rand(batch, 1, generator=g) - 2 * math.pi
        xs.append(x)
        ys.append(torch.sin(x))
    return xs, ys

def stream_continual_A_then_B(stepsA: int, stepsB: int, batch: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)

    xsA, ysA, xsB, ysB = [], [], [], []
    for _ in range(stepsA):
        x = (4 * math.pi) * torch.rand(batch, 1, generator=g) - 2 * math.pi
        xsA.append(x)
        ysA.append(torch.sin(x))

    for _ in range(stepsB):
        x = (4 * math.pi) * torch.rand(batch, 1, generator=g) - 2 * math.pi
        xsB.append(x)
        ysB.append(torch.cos(1.5 * x + 0.3))

    return xsA, ysA, xsB, ysB

@torch.no_grad()
def eval_mean_loss(model, xs, ys, device, loss_fn) -> float:
    model.eval()
    vals = []
    for x, y in zip(xs, ys):
        x = x.to(device)
        y = y.to(device)
        vals.append(float(loss_fn(model(x), y).item()))
    return float(np.mean(vals)) if vals else float("nan")


# Optimiser factory

BASELINE_OPTS = ["SGD", "AdamW", "RMSprop"]
IMP_OPTS = ["Imp", "Imp_disabled", "Imp_shuffled"]
ALL_CONTINUAL_OPTS = ["SGD", "AdamW", "RMSprop", "Imp", "Imp_disabled", "Imp_shuffled", "SGD_matched"]

def make_opt(
    name: str,
    model: nn.Module,
    phi_params: Tuple[float, float, float],
    seed: int,
    *,
    imp_k: float = 1.0,
    sgd_lr: Optional[float] = None,
    record_history: bool = False,
) -> Optimizer:
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=3e-3)
    if name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=1e-3)
    if name == "SGD_matched":
        if sgd_lr is None or (not np.isfinite(sgd_lr)):
            raise ValueError("SGD_matched requires a finite sgd_lr (matched to Imp alpha_mean).")
        return torch.optim.SGD(model.parameters(), lr=float(sgd_lr), momentum=0.9, nesterov=True)

    # Imp family config
    cfg = dict(
        alpha_max=3e-2,
        alpha_min=1e-6,
        k=float(imp_k),
        phi_params=phi_params,
        record_history=bool(record_history),
    )

    if name == "Imp_disabled":
        cfg["phi_mode"] = "disabled"
    elif name == "Imp_shuffled":
        cfg["phi_mode"] = "shuffled"
        rng = np.random.default_rng(seed)
        buf = rng.normal(0, 0.01, size=400).tolist()
        rng.shuffle(buf)
        cfg["shuffled_dloss_buffer"] = buf
    else:
        cfg["phi_mode"] = "normal"

    return ImpOptimiser(model.parameters(), **cfg)


# Experiment records

@dataclass
class ContinualMetrics:
    diverged: float
    A_pre: float
    B_pre: float
    B_post: float
    B_best: float
    A_post: float
    forgetting: float
    A_ckpt_25: float
    A_ckpt_50: float
    A_ckpt_100: float
    matched_lr: float = float("nan")

@dataclass
class StationaryMetrics:
    diverged: float
    A_post: float
    final_train_loss: float
    time_to_thresh: float  # steps to reach threshold, else nan

@dataclass
class CurvePack:
    steps: np.ndarray
    values: np.ndarray


# Runners

def run_stationary_with_curves(
    xsA, ysA,
    *,
    device: str,
    seed: int,
    phi_params: Tuple[float, float, float],
    opt_name: str,
    imp_k: float = 1.0,
    loss_threshold: float = 0.02,
    eval_every: int = 10,
) -> Tuple[StationaryMetrics, Dict[str, CurvePack], Optional[List[Dict[str, float]]]]:
    set_seed(seed)
    loss_fn = nn.MSELoss()
    model = MLP(1, 32, 1).to(device)

    record_history = (opt_name == "Imp")
    opt = make_opt(opt_name, model, phi_params, seed, imp_k=imp_k, record_history=record_history)

    if isinstance(opt, ImpOptimiser):
        opt.reset_alpha_stats()

    # Use the same held-out evaluation set at every checkpoint.
    xs_eval, ys_eval = stream_task_A(steps=100, batch=64, seed=90000 + seed)

    train_losses = []
    eval_losses = []
    eval_steps = []

    time_to_thresh = float("nan")
    diverged = False

    for t, (x, y) in enumerate(zip(xsA, ysA), start=1):
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)

        if not torch.isfinite(loss):
            diverged = True
            break

        loss.backward()
        if isinstance(opt, ImpOptimiser):
            opt.step(loss=loss)
        else:
            opt.step()

        train_losses.append(float(loss.detach().item()))

        if (t % eval_every) == 0:
            ev = eval_mean_loss(model, xs_eval, ys_eval, device, loss_fn)
            eval_steps.append(t)
            eval_losses.append(ev)
            if (not np.isfinite(time_to_thresh)) and (ev <= loss_threshold):
                time_to_thresh = float(t)

    if diverged:
        A_post = float("inf")
        final_train_loss = float("inf")
    else:
        A_post = eval_mean_loss(model, xs_eval, ys_eval, device, loss_fn)
        final_train_loss = float(train_losses[-1]) if train_losses else float("nan")

    metrics = StationaryMetrics(
        diverged=float(diverged),
        A_post=float(A_post),
        final_train_loss=float(final_train_loss),
        time_to_thresh=float(time_to_thresh),
    )

    curves = {
        "train_loss": CurvePack(steps=np.arange(1, len(train_losses) + 1), values=np.array(train_losses, dtype=float)),
        "eval_loss": CurvePack(steps=np.array(eval_steps, dtype=int), values=np.array(eval_losses, dtype=float)),
    }

    imp_hist = None
    if isinstance(opt, ImpOptimiser) and record_history:
        imp_hist = opt.history
        zs = np.array([h["z"] for h in imp_hist], dtype=float)
        alphas = np.array([h["alpha"] for h in imp_hist], dtype=float)
        dloss = np.array([h["dloss"] for h in imp_hist], dtype=float)
        curves["z"] = CurvePack(steps=np.arange(1, len(zs) + 1), values=zs)
        curves["alpha"] = CurvePack(steps=np.arange(1, len(alphas) + 1), values=alphas)
        curves["dloss"] = CurvePack(steps=np.arange(1, len(dloss) + 1), values=dloss)

    return metrics, curves, imp_hist


def run_continual_metrics_only(
    xsA, ysA, xsB, ysB,
    *,
    device: str,
    seed: int,
    phi_params: Tuple[float, float, float],
    imp_k: float,
    sgd_lr_matched: Optional[float],
    opt_name: str,
    b_check_fracs: Tuple[float, float, float] = (0.25, 0.50, 1.00),
) -> Tuple[ContinualMetrics, Dict[str, float]]:
    set_seed(seed)
    loss_fn = nn.MSELoss()
    model = MLP(1, 32, 1).to(device)

    opt = make_opt(opt_name, model, phi_params, seed, imp_k=imp_k, sgd_lr=sgd_lr_matched, record_history=False)

    if isinstance(opt, ImpOptimiser):
        opt.reset_alpha_stats()

    diverged = False

    stepsB_total = len(xsB)
    ckpt_steps = [max(1, int(round(frac * stepsB_total))) for frac in b_check_fracs]
    ckpt_labels = [f"A_ckpt_{int(round(frac*100))}" for frac in b_check_fracs]
    ckpt_vals = {lab: float("nan") for lab in ckpt_labels}
    ckpt_set = set()

    # Train A
    for x, y in zip(xsA, ysA):
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        if isinstance(opt, ImpOptimiser):
            opt.step(loss=loss)
        else:
            opt.step()

    if diverged:
        A_pre = B_pre = B_post = B_best = A_post = float("inf")
        for lab in ckpt_labels:
            ckpt_vals[lab] = float("inf")
    else:
        A_pre = eval_mean_loss(model, xsA, ysA, device, loss_fn)
        B_pre = eval_mean_loss(model, xsB, ysB, device, loss_fn)

        # Train B
        B_best = float("inf")
        for t, (x, y) in enumerate(zip(xsB, ysB), start=1):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            if isinstance(opt, ImpOptimiser):
                opt.step(loss=loss)
            else:
                opt.step()

            lv = float(loss.detach().item())
            if np.isfinite(lv):
                B_best = min(B_best, lv)

            for step_needed, lab in zip(ckpt_steps, ckpt_labels):
                if (t >= step_needed) and (lab not in ckpt_set):
                    ckpt_vals[lab] = eval_mean_loss(model, xsA, ysA, device, loss_fn)
                    ckpt_set.add(lab)

        if diverged:
            B_post = A_post = float("inf")
            for lab in ckpt_labels:
                if not np.isfinite(ckpt_vals.get(lab, float("nan"))):
                    ckpt_vals[lab] = float("inf")
        else:
            B_post = eval_mean_loss(model, xsB, ysB, device, loss_fn)
            A_post = eval_mean_loss(model, xsA, ysA, device, loss_fn)
            for lab in ckpt_labels:
                if not np.isfinite(ckpt_vals.get(lab, float("nan"))):
                    ckpt_vals[lab] = eval_mean_loss(model, xsA, ysA, device, loss_fn)

    alpha_stats = {"alpha_mean": float("nan"), "alpha_min": float("nan"), "alpha_max": float("nan")}
    if isinstance(opt, ImpOptimiser):
        alpha_stats = opt.alpha_stats()

    m = ContinualMetrics(
        diverged=float(diverged),
        A_pre=float(A_pre),
        B_pre=float(B_pre),
        B_post=float(B_post),
        B_best=float(B_best),
        A_post=float(A_post),
        forgetting=float(A_post - A_pre),
        A_ckpt_25=float(ckpt_vals.get("A_ckpt_25", float("nan"))),
        A_ckpt_50=float(ckpt_vals.get("A_ckpt_50", float("nan"))),
        A_ckpt_100=float(ckpt_vals.get("A_ckpt_100", float("nan"))),
        matched_lr=float("nan"),
    )
    return m, alpha_stats


def run_continual_with_eval_curves(
    xsA, ysA, xsB, ysB,
    *,
    device: str,
    seed: int,
    phi_params: Tuple[float, float, float],
    opt_name: str,
    imp_k: float,
    sgd_lr_matched: Optional[float],
    eval_every: int = 10,
) -> Dict[str, CurvePack]:
    set_seed(seed)
    loss_fn = nn.MSELoss()
    model = MLP(1, 32, 1).to(device)
    opt = make_opt(opt_name, model, phi_params, seed, imp_k=imp_k, sgd_lr=sgd_lr_matched, record_history=False)

    # Use fixed held-out evaluation sets for Tasks A and B.
    xsA_eval, ysA_eval = stream_task_A(steps=120, batch=64, seed=91000 + seed)

    g = torch.Generator()
    g.manual_seed(92000 + seed)
    xsB_eval, ysB_eval = [], []
    for _ in range(120):
        x = (4 * math.pi) * torch.rand(64, 1, generator=g) - 2 * math.pi
        xsB_eval.append(x)
        ysB_eval.append(torch.cos(1.5 * x + 0.3))

    A_curve_steps, A_curve_vals = [], []
    B_curve_steps, B_curve_vals = [], []

    def record(step: int):
        A_curve_steps.append(step)
        A_curve_vals.append(eval_mean_loss(model, xsA_eval, ysA_eval, device, loss_fn))
        B_curve_steps.append(step)
        B_curve_vals.append(eval_mean_loss(model, xsB_eval, ysB_eval, device, loss_fn))

    step_counter = 0

    # Train A
    for x, y in zip(xsA, ysA):
        step_counter += 1
        x = x.to(device); y = y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            break
        loss.backward()
        if isinstance(opt, ImpOptimiser):
            opt.step(loss=loss)
        else:
            opt.step()
        if (step_counter % eval_every) == 0:
            record(step_counter)

    # Train B
    for x, y in zip(xsB, ysB):
        step_counter += 1
        x = x.to(device); y = y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            break
        loss.backward()
        if isinstance(opt, ImpOptimiser):
            opt.step(loss=loss)
        else:
            opt.step()
        if (step_counter % eval_every) == 0:
            record(step_counter)

    curves = {
        "A_eval_loss": CurvePack(steps=np.array(A_curve_steps, dtype=int), values=np.array(A_curve_vals, dtype=float)),
        "B_eval_loss": CurvePack(steps=np.array(B_curve_steps, dtype=int), values=np.array(B_curve_vals, dtype=float)),
    }
    return curves


# Plotting helpers

def save_fig(path_pdf: str, also_png: bool = True) -> None:
    plt.tight_layout()
    plt.savefig(path_pdf, bbox_inches="tight")
    if also_png:
        path_png = os.path.splitext(path_pdf)[0] + ".png"
        plt.savefig(path_png, bbox_inches="tight", dpi=200)
    plt.close()

def plot_impedance_mapping_controls(phi_params: Tuple[float, float, float], outpath: str, *, no_titles: bool) -> None:
    xs = np.linspace(-0.5, 0.5, 600)
    y_h = phi_curve(xs, phi_params)
    y_l = phi_linear(xs, slope=max(1e-6, (np.max(np.abs(y_h)) / max(1e-6, np.max(np.abs(xs)))) * 0.5))
    y_f = phi_flat(xs)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(xs, y_h, label="Human φ (anchored sigmoid)")
    plt.plot(xs, y_l, label="Linear φ (control)")
    plt.plot(xs, y_f, label="Flat φ (control)")
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel(r"$\Delta \ell$")
    plt.ylabel(r"$\phi(\Delta \ell)$")
    if not no_titles:
        plt.title("Impedance mapping φ (human prior vs controls)")
    plt.legend(frameon=False)
    save_fig(outpath)

def plot_imp_dynamics(imp_hist: List[Dict[str, float]], outpath: str, *, no_titles: bool) -> None:
    if not imp_hist:
        print("Warning: no optimiser trajectory is available for the dynamics plot.")
        return

    loss = np.array([h["loss"] for h in imp_hist], dtype=float)
    z = np.array([h["z"] for h in imp_hist], dtype=float)
    alpha = np.array([h["alpha"] for h in imp_hist], dtype=float)
    dloss = np.array([h["dloss"] for h in imp_hist], dtype=float)
    steps = np.arange(1, len(loss) + 1)

    plt.figure(figsize=(8.0, 7.0))

    plt.subplot(4, 1, 1)
    plt.plot(steps, loss)
    plt.ylabel("train loss")
    if not no_titles:
        plt.title("Imp internal dynamics (one representative run)")

    plt.subplot(4, 1, 2)
    plt.plot(steps, dloss)
    plt.axhline(0, linewidth=0.8)
    plt.ylabel("Δloss (EMA)")

    plt.subplot(4, 1, 3)
    plt.plot(steps, z)
    plt.ylabel("z (impedance)")

    plt.subplot(4, 1, 4)
    plt.plot(steps, alpha)
    plt.ylabel("α (effective)")
    plt.xlabel("training step")

    save_fig(outpath)

def plot_mean_curve_with_band(ax, xs_list: List[np.ndarray], ys_list: List[np.ndarray], label: str):
    min_len = min(len(x) for x in xs_list)
    X = xs_list[0][:min_len]
    Y = np.stack([y[:min_len] for y in ys_list], axis=0)
    m, s = mean_std(Y, axis=0)
    ax.plot(X, m, label=label)
    ax.fill_between(X, m - s, m + s, alpha=0.15)

def plot_stationary_curves(
    stationary_curves: Dict[str, Dict[int, Dict[str, CurvePack]]],
    outpath: str,
    *,
    no_titles: bool,
) -> None:
    plt.figure(figsize=(7.6, 4.6))
    ax = plt.gca()

    for opt in ["AdamW", "SGD", "Imp"]:
        xs_list, ys_list = [], []
        for seed, packs in stationary_curves.get(opt, {}).items():
            if "eval_loss" not in packs:
                continue
            xs_list.append(packs["eval_loss"].steps.astype(int))
            ys_list.append(packs["eval_loss"].values.astype(float))
        if xs_list:
            plot_mean_curve_with_band(ax, xs_list, ys_list, label=opt)

    ax.set_xlabel("training step")
    ax.set_ylabel("eval loss on A")
    if not no_titles:
        ax.set_title("Stationary task: conservatism costs convergence speed")
    ax.legend(frameon=False)
    save_fig(outpath)

def plot_continual_curves(
    continual_curves: Dict[str, Dict[int, Dict[str, CurvePack]]],
    switch_step: int,
    outpath: str,
    *,
    no_titles: bool,
) -> None:
    plt.figure(figsize=(8.2, 5.8))

    ax1 = plt.subplot(2, 1, 1)
    for opt in ["AdamW", "Imp", "Imp_disabled", "Imp_shuffled", "SGD_matched"]:
        xs_list, ys_list = [], []
        for seed, packs in continual_curves.get(opt, {}).items():
            if "A_eval_loss" not in packs:
                continue
            xs_list.append(packs["A_eval_loss"].steps.astype(int))
            ys_list.append(packs["A_eval_loss"].values.astype(float))
        if xs_list:
            plot_mean_curve_with_band(ax1, xs_list, ys_list, label=opt)
    ax1.axvline(switch_step, linestyle="--", linewidth=1.0)
    ax1.set_ylabel("A eval loss")
    if not no_titles:
        ax1.set_title("Continual A→B: retention/interference (A)")

    ax2 = plt.subplot(2, 1, 2)
    for opt in ["AdamW", "Imp", "Imp_disabled", "Imp_shuffled", "SGD_matched"]:
        xs_list, ys_list = [], []
        for seed, packs in continual_curves.get(opt, {}).items():
            if "B_eval_loss" not in packs:
                continue
            xs_list.append(packs["B_eval_loss"].steps.astype(int))
            ys_list.append(packs["B_eval_loss"].values.astype(float))
        if xs_list:
            plot_mean_curve_with_band(ax2, xs_list, ys_list, label=opt)
    ax2.axvline(switch_step, linestyle="--", linewidth=1.0)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("B eval loss")
    if not no_titles:
        ax2.set_title("B acquisition (B)")
    ax2.legend(frameon=False)

    save_fig(outpath)

def plot_gain_tradeoff(k_rows: Dict[float, Dict[str, float]], outpath: str, *, no_titles: bool) -> None:
    ks = np.array(sorted(k_rows.keys()), dtype=float)
    forgetting = np.array([safe_float(k_rows[k].get("Imp_forgetting", np.nan)) for k in ks], dtype=float)
    bbest = np.array([safe_float(k_rows[k].get("Imp_B_best", np.nan)) for k in ks], dtype=float)

    plt.figure(figsize=(7.2, 4.6))
    ax = plt.gca()
    ax.plot(ks, forgetting, marker="o", label="Forgetting (A_post - A_pre)")
    ax.plot(ks, bbest, marker="o", label="B_best (plasticity proxy)")
    ax.set_xscale("log")
    ax.set_xlabel("k (gain, log scale)")
    ax.set_ylabel("metric value")
    if not no_titles:
        ax.set_title("Stability–plasticity trade-off across k")
    ax.legend(frameon=False)
    save_fig(outpath)


# CSV writer

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    if np.isfinite(v):
                        vals.append(f"{v:.10g}")
                    else:
                        vals.append("nan")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


# Main

def load_phi_params(args) -> Tuple[Tuple[float, float, float], str]:
    if args.phi is not None:
        phi_params = tuple(float(v) for v in args.phi)
        return phi_params, "--phi override"
    popt = np.load(args.phi_params_path)
    popt = np.asarray(popt, dtype=float).reshape(-1)
    if popt.size != 3:
        raise ValueError(f"Expected phi params file to contain 3 values [Apos, mu, s], got {popt.shape}")
    phi_params = (float(popt[0]), float(popt[1]), float(popt[2]))
    return phi_params, Path(args.phi_params_path).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval_every", type=int, default=10)

    # φ config
    ap.add_argument(
        "--phi_params_path",
        default=str(Path(__file__).resolve().with_name("20261127_ORE_imp_opt_params.npy")),
        help="Parameter file (defaults to the archived vector beside this script)",
    )
    ap.add_argument("--phi", type=float, nargs=3, default=None)

    # Imp config
    ap.add_argument("--k_default", type=float, default=1.0)
    ap.add_argument("--k_sweep", type=float, nargs="*", default=[0.1, 0.3, 1.0, 3.0])

    # Stationary threshold (for time-to-threshold)
    ap.add_argument("--stationary_thresh", type=float, default=0.02)

    # Output controls
    ap.add_argument("--out_root", type=str, help="Base output folder (defaults to the script folder)")
    ap.add_argument("--run_id", type=str, default=None, help="Run identifier; if omitted uses timestamp")
    ap.add_argument("--no_titles", action="store_true", help="Suppress plot titles")

    args = ap.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root).resolve() if args.out_root is not None else Path(__file__).resolve().parent
    results_dir = out_root / "results" / run_id
    figures_dir = out_root / "figures" / run_id
    ensure_dir(results_dir)
    ensure_dir(figures_dir)

    phi_params, phi_source = load_phi_params(args)
    print("\n--- φ configuration ---")
    print(f"phi params [Apos, mu, s] = {phi_params}")
    print(f"phi source = {phi_source}")
    print("-----------------------\n")

    print(f"--- output ---")
    print(f"run_id      = {run_id}")
    print(f"results_dir = {results_dir}")
    print(f"figures_dir = {figures_dir}")
    print(f"no_titles   = {bool(args.no_titles)}")
    print("------------\n")

    device = args.device
    steps_total = int(args.steps)
    stepsA = steps_total // 2
    stepsB = steps_total - stepsA
    switch_step = stepsA

    # Plot the fitted mapping and the ablation controls.

    plot_impedance_mapping_controls(phi_params, outpath=os.path.join(figures_dir, "impedance_mapping_controls.pdf"), no_titles=bool(args.no_titles))

    # Stationary learning benchmark
    stationary_metrics_rows: List[Dict[str, Any]] = []
    stationary_curves: Dict[str, Dict[int, Dict[str, CurvePack]]] = {opt: {} for opt in ["SGD", "AdamW", "Imp"]}

    for seed in args.seeds:
        xsA_stream, ysA_stream = stream_task_A(steps=steps_total, batch=args.batch_size, seed=5000 + seed)

        for opt in ["SGD", "AdamW", "Imp"]:
            m, curves, imp_hist = run_stationary_with_curves(
                xsA_stream, ysA_stream,
                device=device,
                seed=seed,
                phi_params=phi_params,
                opt_name=opt,
                imp_k=float(args.k_default),
                loss_threshold=float(args.stationary_thresh),
                eval_every=int(args.eval_every),
            )

            stationary_curves[opt][seed] = curves

            stationary_metrics_rows.append({
                "seed": seed,
                "opt": opt,
                "diverged": m.diverged,
                "A_post": m.A_post,
                "final_train_loss": m.final_train_loss,
                "time_to_thresh": m.time_to_thresh,
            })

            # Use the first seed for the example optimiser trajectory.
            if (opt == "Imp") and (seed == args.seeds[0]) and (imp_hist is not None):
                plot_imp_dynamics(
                    imp_hist,
                    outpath=os.path.join(figures_dir, "fig1_dynamics_imp.pdf"),
                    no_titles=bool(args.no_titles),
                )

    write_csv(os.path.join(results_dir, "stationary_metrics.csv"), stationary_metrics_rows)
    plot_stationary_curves(
        stationary_curves,
        outpath=os.path.join(figures_dir, "fig2_stationary_curves.pdf"),
        no_titles=bool(args.no_titles),
    )

    # Continual learning benchmark and ablations
    continual_metrics_rows: List[Dict[str, Any]] = []
    continual_curves: Dict[str, Dict[int, Dict[str, CurvePack]]] = {
        opt: {} for opt in ["AdamW", "Imp", "Imp_disabled", "Imp_shuffled", "SGD_matched"]
    }

    imp_alpha_stats_per_seed: Dict[int, Dict[str, float]] = {}

    for seed in args.seeds:
        xsA, ysA, xsB, ysB = stream_continual_A_then_B(stepsA, stepsB, args.batch_size, seed=6000 + seed)

        # Imp baseline (for matching)
        m_imp, alpha_stats = run_continual_metrics_only(
            xsA, ysA, xsB, ysB,
            device=device,
            seed=seed,
            phi_params=phi_params,
            imp_k=float(args.k_default),
            sgd_lr_matched=None,
            opt_name="Imp",
        )
        imp_alpha_stats_per_seed[seed] = alpha_stats
        alpha_mean = safe_float(alpha_stats.get("alpha_mean", np.nan))
        if not np.isfinite(alpha_mean):
            raise RuntimeError("Imp alpha_mean is not finite; cannot run SGD_matched reliably.")

        continual_metrics_rows.append({**m_imp.__dict__, "seed": seed, "opt": "Imp", "matched_lr": float("nan")})

        # Baselines + ablations (including SGD_matched)
        for opt in ["AdamW", "Imp_disabled", "Imp_shuffled", "SGD_matched"]:
            if opt == "SGD_matched":
                sgd_lr = alpha_mean
            else:
                sgd_lr = None

            m, _ = run_continual_metrics_only(
                xsA, ysA, xsB, ysB,
                device=device,
                seed=seed,
                phi_params=phi_params,
                imp_k=float(args.k_default),
                sgd_lr_matched=sgd_lr,
                opt_name=opt,
            )
            row = {
                **m.__dict__,
                "seed": seed,
                "opt": opt,
                "matched_lr": (float(sgd_lr) if sgd_lr is not None else float("nan")),
            }
            continual_metrics_rows.append(row)

        continual_curves["Imp"][seed] = run_continual_with_eval_curves(
            xsA, ysA, xsB, ysB,
            device=device,
            seed=seed,
            phi_params=phi_params,
            opt_name="Imp",
            imp_k=float(args.k_default),
            sgd_lr_matched=None,
            eval_every=int(args.eval_every),
        )
        continual_curves["AdamW"][seed] = run_continual_with_eval_curves(
            xsA, ysA, xsB, ysB,
            device=device,
            seed=seed,
            phi_params=phi_params,
            opt_name="AdamW",
            imp_k=float(args.k_default),
            sgd_lr_matched=None,
            eval_every=int(args.eval_every),
        )
        continual_curves["Imp_disabled"][seed] = run_continual_with_eval_curves(
            xsA, ysA, xsB, ysB,
            device=device,
            seed=seed,
            phi_params=phi_params,
            opt_name="Imp_disabled",
            imp_k=float(args.k_default),
            sgd_lr_matched=None,
            eval_every=int(args.eval_every),
        )
        continual_curves["Imp_shuffled"][seed] = run_continual_with_eval_curves(
            xsA, ysA, xsB, ysB,
            device=device,
            seed=seed,
            phi_params=phi_params,
            opt_name="Imp_shuffled",
            imp_k=float(args.k_default),
            sgd_lr_matched=None,
            eval_every=int(args.eval_every),
        )
        continual_curves["SGD_matched"][seed] = run_continual_with_eval_curves(
            xsA, ysA, xsB, ysB,
            device=device,
            seed=seed,
            phi_params=phi_params,
            opt_name="SGD_matched",
            imp_k=float(args.k_default),
            sgd_lr_matched=alpha_mean,
            eval_every=int(args.eval_every),
        )

    write_csv(os.path.join(results_dir, "continual_metrics.csv"), continual_metrics_rows)
    plot_continual_curves(
        continual_curves,
        switch_step=switch_step,
        outpath=os.path.join(figures_dir, "fig3_continual_curves.pdf"),
        no_titles=bool(args.no_titles),
    )

    # Evaluate the gain sweep.
    k_rows: Dict[float, Dict[str, float]] = {}

    for k_val in [float(k) for k in args.k_sweep]:
        per_seed_imp = []
        per_seed_alpha = []

        for seed in args.seeds:
            xsA, ysA, xsB, ysB = stream_continual_A_then_B(stepsA, stepsB, args.batch_size, seed=6000 + seed)

            m_imp, alpha_stats = run_continual_metrics_only(
                xsA, ysA, xsB, ysB,
                device=device,
                seed=seed,
                phi_params=phi_params,
                imp_k=k_val,
                sgd_lr_matched=None,
                opt_name="Imp",
            )
            per_seed_imp.append(m_imp)
            per_seed_alpha.append(alpha_stats)

        imp_forgetting = float(np.mean([m.forgetting for m in per_seed_imp]))
        imp_bbest = float(np.mean([m.B_best for m in per_seed_imp]))
        alpha_mean = float(np.mean([safe_float(a.get("alpha_mean", np.nan)) for a in per_seed_alpha]))

        k_rows[k_val] = {
            "Imp_forgetting": imp_forgetting,
            "Imp_B_best": imp_bbest,
            "Imp_alpha_mean": alpha_mean,
        }

    k_csv_rows = [{"k": float(k), **vals} for k, vals in sorted(k_rows.items(), key=lambda kv: kv[0])]
    write_csv(os.path.join(results_dir, "k_sweep.csv"), k_csv_rows)

    plot_gain_tradeoff(
        k_rows,
        outpath=os.path.join(figures_dir, "gain_tradeoff.pdf"),
        no_titles=bool(args.no_titles),
    )

    # Save the run configuration and summary statistics.
    
    out = {
        "config": {
            "device": device,
            "steps_total": steps_total,
            "stepsA": stepsA,
            "stepsB": stepsB,
            "batch_size": int(args.batch_size),
            "seeds": list(args.seeds),
            "eval_every": int(args.eval_every),
            "phi_params": [float(x) for x in phi_params],
            "phi_source": str(phi_source),
            "k_default": float(args.k_default),
            "k_sweep": [float(x) for x in args.k_sweep],
            "stationary_threshold": float(args.stationary_thresh),
            "run_id": run_id,
            "no_titles": bool(args.no_titles),
            "out_root": args.out_root if args.out_root is not None else ".",
        },
        "imp_alpha_stats_per_seed": imp_alpha_stats_per_seed,
        "k_sweep_summary": {str(k): {kk: float(vv) for kk, vv in vals.items()} for k, vals in k_rows.items()},
    }
    with open(os.path.join(results_dir, "imp_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nDONE.")
    print("Wrote:")
    print(f"  {os.path.join(results_dir, 'imp_results.json')}")
    print(f"  {os.path.join(results_dir, 'stationary_metrics.csv')}")
    print(f"  {os.path.join(results_dir, 'continual_metrics.csv')}")
    print(f"  {os.path.join(results_dir, 'k_sweep.csv')}")
    print("Saved figures:")
    print(f"  {os.path.join(figures_dir, 'impedance_mapping_controls.(pdf/png)')}")
    print(f"  {os.path.join(figures_dir, 'fig1_dynamics_imp.(pdf/png)')}")
    print(f"  {os.path.join(figures_dir, 'fig2_stationary_curves.(pdf/png)')}")
    print(f"  {os.path.join(figures_dir, 'fig3_continual_curves.(pdf/png)')}")
    print(f"  {os.path.join(figures_dir, 'gain_tradeoff.(pdf/png)')}")


if __name__ == "__main__":
    main()
