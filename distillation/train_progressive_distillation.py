"""Train a 4-to-1 progressively distilled conditional diffusion student.

Each round makes a student reproduce four deterministic DDIM teacher steps in
one step. The resulting checkpoint is directly consumable by
``distilled_dps_inference_ddp.py``.
"""
import os
import sys
import copy
import math
import argparse
import logging
import random
import socket
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.dataloader_accelerate_miroc6 import get_dataset
from distillation.distilled_sampler import get_named_beta_schedule
from distillation.model import DiffusersUNetWrapper, load_yaml


# ============================================================
# Utils
# ============================================================
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def setup_seed(seed, rank):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process(rank):
    return rank == 0


def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def get_epochs_for_round(epochs_per_round, round_idx):
    if isinstance(epochs_per_round, int):
        return epochs_per_round
    elif isinstance(epochs_per_round, (list, tuple)):
        if round_idx >= len(epochs_per_round):
            raise ValueError(
                f"round_idx={round_idx} exceeds epochs_per_round length={len(epochs_per_round)}"
            )
        return int(epochs_per_round[round_idx])
    else:
        raise TypeError(
            f"epochs_per_round must be int or list/tuple, got {type(epochs_per_round)}"
        )


def detect_latest_completed_round(save_dir, num_rounds):
    latest = -1
    for r in range(num_rounds):
        ckpt_path = os.path.join(save_dir, f"student_round{r}_final.pt")
        if os.path.exists(ckpt_path):
            latest = r
        else:
            break
    return latest


def get_steps_after_rounds(initial_steps, completed_rounds):
    steps = initial_steps
    for _ in range(completed_rounds):
        steps = steps // 4
    return steps


# ============================================================
# Logging
# ============================================================
def setup_logger(save_dir, rank):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(f"distillation_spawn_amp_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][Rank %(rank)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    class RankFilter(logging.Filter):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank

        def filter(self, record):
            record.rank = self.rank
            return True

    rank_filter = RankFilter(rank)

    fh = logging.FileHandler(os.path.join(save_dir, f"distill_rank{rank}.log"))
    fh.setFormatter(fmt)
    fh.addFilter(rank_filter)
    logger.addHandler(fh)

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.addFilter(rank_filter)
        logger.addHandler(ch)

    return logger


# ============================================================
# Schedule utils
# ============================================================
def build_alpha_schedule(noise_schedule="linear", num_steps=1000):
    betas = get_named_beta_schedule(noise_schedule, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas_cumprod


def space_timesteps(num_timesteps, section_counts):
    """Return the DDIM-spaced indices used by the original teacher schedule."""
    if isinstance(section_counts, str) and section_counts.startswith('ddim'):
        desired_count = int(section_counts[len('ddim'):])
        for stride in range(1, num_timesteps):
            steps = range(0, num_timesteps, stride)
            if len(steps) == desired_count:
                return set(steps)
        raise ValueError(f'Cannot create exactly {desired_count} DDIM steps from {num_timesteps}.')
    raise ValueError('Only a single ddimN timestep specification is supported.')


def build_ddim_timesteps(total_steps, ddim_steps):
    use_timesteps = space_timesteps(total_steps, f"ddim{ddim_steps}")
    timesteps = sorted(list(use_timesteps))
    return np.array(timesteps, dtype=np.int64)


def build_timestep_quads(teacher_timesteps, student_timesteps):
    """Build 4-to-1 timestep tuples for progressive distillation.

    The teacher traverses four transitions between adjacent student timesteps.
    Three approximately equally spaced intermediate teacher timesteps are used.
    """
    quads = []
    for i in range(len(student_timesteps) - 1):
        t_end = student_timesteps[i]
        t_start = student_timesteps[i + 1]
        teacher_between = sorted([t for t in teacher_timesteps if t_end < t < t_start])
        if len(teacher_between) < 3:
            continue
        # Choose three approximately equally spaced intermediate timesteps.
        n = len(teacher_between)
        idx1 = n * 3 // 4  # Nearest to t_start.
        idx2 = n * 2 // 4  # Midpoint.
        idx3 = n * 1 // 4  # Nearest to t_end.
        t_mid1 = teacher_between[idx1]
        t_mid2 = teacher_between[idx2]
        t_mid3 = teacher_between[idx3]
        quads.append((int(t_start), int(t_mid1), int(t_mid2), int(t_mid3), int(t_end)))
    return quads


def find_anchor_timestep(student_timesteps, t_start, max_back_steps=1):
    student_timesteps = list(student_timesteps)
    if t_start not in student_timesteps:
        return None
    idx = student_timesteps.index(t_start)
    anchor_idx = min(idx + max_back_steps, len(student_timesteps) - 1)
    if anchor_idx == idx:
        return None
    return int(student_timesteps[anchor_idx])


# ============================================================
# DDIM
# ============================================================
def ddim_step(x_t, eps_pred, alphas_cumprod, t_now, t_next):
    device = x_t.device
    dtype = x_t.dtype

    alpha_bar_now = torch.tensor(alphas_cumprod[t_now], device=device, dtype=dtype)
    if t_next >= 0:
        alpha_bar_next = torch.tensor(alphas_cumprod[t_next], device=device, dtype=dtype)
    else:
        alpha_bar_next = torch.tensor(1.0, device=device, dtype=dtype)

    x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_now) * eps_pred) / torch.sqrt(alpha_bar_now)
    x_next = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * eps_pred
    return x_next, x0_pred


def model_forward(model, x_t, t_idx, fixed_condition):
    B = x_t.shape[0]
    C = x_t.shape[1]
    device = x_t.device
    t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
    model_output = model(x_t, t_tensor, fixed_condition=fixed_condition)
    if model_output.shape[1] == 2 * C:
        eps_pred = model_output[:, :C]
    else:
        eps_pred = model_output
    return eps_pred


# ============================================================
# Teacher / Student transitions
# ============================================================
@torch.no_grad()
def teacher_four_steps(teacher, x_t, t_start, t_mid1, t_mid2, t_mid3, t_end, fixed_condition, alphas_cumprod, use_fp16=False):
    with autocast("cuda", enabled=use_fp16):
        eps_1 = model_forward(teacher, x_t, t_start, fixed_condition)
        x_1, _ = ddim_step(x_t, eps_1, alphas_cumprod, t_start, t_mid1)

        eps_2 = model_forward(teacher, x_1, t_mid1, fixed_condition)
        x_2, _ = ddim_step(x_1, eps_2, alphas_cumprod, t_mid1, t_mid2)

        eps_3 = model_forward(teacher, x_2, t_mid2, fixed_condition)
        x_3, _ = ddim_step(x_2, eps_3, alphas_cumprod, t_mid2, t_mid3)

        eps_4 = model_forward(teacher, x_3, t_mid3, fixed_condition)
        x_target, x0_target = ddim_step(x_3, eps_4, alphas_cumprod, t_mid3, t_end)
    return x_target, x0_target


@torch.no_grad()
def rollout_model_steps(model, x_anchor, t_anchor, t_target, timesteps, fixed_condition, alphas_cumprod, use_fp16=False):
    if t_anchor == t_target:
        return x_anchor

    timesteps = list(timesteps)
    idx_anchor = timesteps.index(t_anchor)
    idx_target = timesteps.index(t_target)
    x = x_anchor

    with autocast("cuda", enabled=use_fp16):
        for idx in range(idx_anchor, idx_target, -1):
            t_now = timesteps[idx]
            t_next = timesteps[idx - 1]
            eps = model_forward(model, x, t_now, fixed_condition)
            x, _ = ddim_step(x, eps, alphas_cumprod, t_now, t_next)
    return x


# ============================================================
# State source sampling
# ============================================================
def sample_forward_noised_state(x0, t_start, alphas_cumprod):
    noise = torch.randn_like(x0)
    sqrt_alpha = math.sqrt(alphas_cumprod[t_start])
    sqrt_one_minus_alpha = math.sqrt(1.0 - alphas_cumprod[t_start])
    x_t = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    return x_t


@torch.no_grad()
def sample_teacher_rollout_state(
    teacher, x0, cond, t_start, student_timesteps, alphas_cumprod,
    rollout_back_steps=1, use_fp16=False
):
    anchor = find_anchor_timestep(student_timesteps, t_start, max_back_steps=rollout_back_steps)
    if anchor is None:
        return sample_forward_noised_state(x0, t_start, alphas_cumprod), "forward_fallback"

    x_anchor = sample_forward_noised_state(x0, anchor, alphas_cumprod)
    x_t = rollout_model_steps(
        teacher, x_anchor, anchor, t_start, student_timesteps, cond, alphas_cumprod, use_fp16=use_fp16
    )
    return x_t, "teacher_rollout"


@torch.no_grad()
def sample_student_rollout_state(
    student, x0, cond, t_start, student_timesteps, alphas_cumprod,
    rollout_back_steps=1, use_fp16=False
):
    anchor = find_anchor_timestep(student_timesteps, t_start, max_back_steps=rollout_back_steps)
    if anchor is None:
        return sample_forward_noised_state(x0, t_start, alphas_cumprod), "forward_fallback"

    x_anchor = sample_forward_noised_state(x0, anchor, alphas_cumprod)
    x_t = rollout_model_steps(
        student, x_anchor, anchor, t_start, student_timesteps, cond, alphas_cumprod, use_fp16=use_fp16
    )
    return x_t, "student_rollout"


def get_source_mix_probs(round_idx, epoch_idx, total_epochs, cfg):
    mix_cfg = cfg.get("state_source_mixing", {})
    use_dynamic = mix_cfg.get("use_dynamic_schedule", True)

    if not use_dynamic:
        p_forward = mix_cfg.get("p_forward", 0.7)
        p_teacher = mix_cfg.get("p_teacher", 0.1)
        p_student = mix_cfg.get("p_student", 0.2)
    else:
        progress = (epoch_idx + 1) / max(total_epochs, 1)
        if round_idx == 0:
            p_forward = 0.75
            p_teacher = 0.15
            p_student = 0.10
        elif round_idx == 1:
            p_forward = 0.55
            p_teacher = 0.15
            p_student = 0.30
        else:
            p_forward = 0.40
            p_teacher = 0.10
            p_student = 0.50

        delta = 0.25 * progress
        shift = min(delta, p_forward - 0.20) if p_forward > 0.20 else 0.0
        p_forward -= shift
        p_student += shift

    total = p_forward + p_teacher + p_student
    return {
        "forward": p_forward / total,
        "teacher": p_teacher / total,
        "student": p_student / total,
    }


def sample_state_source(source_probs):
    r = random.random()
    if r < source_probs["forward"]:
        return "forward"
    elif r < source_probs["forward"] + source_probs["teacher"]:
        return "teacher"
    else:
        return "student"


# ============================================================
# Pair sampler
# ============================================================
class PairPrioritySampler:
    def __init__(self, pairs, alpha=0.6, eps=1e-3, ema_decay=0.9, warmup_uniform_epochs=1):
        self.pairs = pairs
        self.alpha = alpha
        self.eps = eps
        self.ema_decay = ema_decay
        self.warmup_uniform_epochs = warmup_uniform_epochs
        self.ema_losses = np.ones(len(pairs), dtype=np.float64)

    def sample(self, epoch_idx):
        if len(self.pairs) == 1 or epoch_idx < self.warmup_uniform_epochs:
            idx = np.random.randint(0, len(self.pairs))
            return idx, self.pairs[idx]

        priorities = (self.ema_losses + self.eps) ** self.alpha
        probs = priorities / priorities.sum()
        idx = np.random.choice(len(self.pairs), p=probs)
        return idx, self.pairs[idx]

    def update(self, idx, loss_value):
        self.ema_losses[idx] = self.ema_decay * self.ema_losses[idx] + (1.0 - self.ema_decay) * float(loss_value)

    def stats(self):
        return {
            "min": float(self.ema_losses.min()),
            "max": float(self.ema_losses.max()),
            "mean": float(self.ema_losses.mean()),
        }


# ============================================================
# Extra consistency losses
# ============================================================
def gradient_consistency_loss(x_pred, x_ref):
    """
    Match spatial gradients of student clean prediction to reference clean field.
    x_pred, x_ref: (B, C, H, W)
    """
    gx_pred = x_pred[..., 1:, :] - x_pred[..., :-1, :]
    gy_pred = x_pred[..., :, 1:] - x_pred[..., :, :-1]

    gx_ref = x_ref[..., 1:, :] - x_ref[..., :-1, :]
    gy_ref = x_ref[..., :, 1:] - x_ref[..., :, :-1]

    loss_gx = F.mse_loss(gx_pred.float(), gx_ref.float())
    loss_gy = F.mse_loss(gy_pred.float(), gy_ref.float())
    return loss_gx + loss_gy


def spectral_consistency_loss(x_pred, x_ref, eps=1e-6):
    """
    Match log-amplitude spectra using 2D FFT.
    x_pred, x_ref: (B, C, H, W)
    """
    fft_pred = torch.fft.rfft2(x_pred.float(), norm="ortho")
    fft_ref = torch.fft.rfft2(x_ref.float(), norm="ortho")

    amp_pred = torch.abs(fft_pred)
    amp_ref = torch.abs(fft_ref)

    log_amp_pred = torch.log(amp_pred + eps)
    log_amp_ref = torch.log(amp_ref + eps)

    return F.mse_loss(log_amp_pred, log_amp_ref)


# ============================================================
# Loss
# ============================================================
def compute_distill_loss(x_pred, x_target, x0_pred, x0_target, cfg):
    loss_cfg = cfg.get("loss", {})

    lambda_x = loss_cfg.get("lambda_x", 1.0)
    lambda_x0 = loss_cfg.get("lambda_x0", 1.0)
    lambda_grad = loss_cfg.get("lambda_grad", 0.0)
    lambda_spec = loss_cfg.get("lambda_spec", 0.0)

    # Base distillation losses
    loss_x = F.mse_loss(x_pred.float(), x_target.float())
    loss_x0 = F.mse_loss(x0_pred.float(), x0_target.float())

    # Extra structural consistency losses (on clean prediction)
    loss_grad = gradient_consistency_loss(x0_pred, x0_target) if lambda_grad > 0 else x0_pred.new_tensor(0.0)
    loss_spec = spectral_consistency_loss(x0_pred, x0_target) if lambda_spec > 0 else x0_pred.new_tensor(0.0)

    total = (
        lambda_x * loss_x
        + lambda_x0 * loss_x0
        + lambda_grad * loss_grad
        + lambda_spec * loss_spec
    )

    return total, {
        "loss_x": loss_x.detach(),
        "loss_x0": loss_x0.detach(),
        "loss_grad": loss_grad.detach(),
        "loss_spec": loss_spec.detach(),
    }


# ============================================================
# Model loading
# ============================================================
def load_teacher_model(model_config_path, checkpoint_path, device, logger, rank):
    model_config = load_yaml(model_config_path)
    model = DiffusersUNetWrapper(model_config, device)

    logger.info(f"Loading teacher checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    mapping = {
        ".query.": ".to_q.",
        ".key.": ".to_k.",
        ".value.": ".to_v.",
        ".proj_attn.": ".to_out.0.",
    }
    new_state_dict = {}
    for name, param in state_dict.items():
        new_name = name
        if "attentions" in name:
            for old, new in mapping.items():
                new_name = new_name.replace(old, new)
        new_state_dict[new_name] = param

    try:
        model.unet.load_state_dict(new_state_dict, strict=True)
        logger.info("Teacher weights loaded (strict=True).")
    except RuntimeError as e:
        logger.warning(f"strict=True failed, retrying with strict=False: {e}")
        model.unet.load_state_dict(new_state_dict, strict=False)

    for module in model.modules():
        if isinstance(module, torch.nn.GroupNorm) and isinstance(module.eps, str):
            module.eps = float(module.eps)

    model.to(device)
    model.eval()
    return model


def save_student_checkpoint(student_ddp, save_path, payload, rank, scaler=None):
    if rank != 0:
        return
    model_to_save = unwrap_model(student_ddp)
    ckpt = {"state_dict": model_to_save.unet.state_dict()}
    ckpt.update(payload)
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, save_path)


# ============================================================
# Distill one round
# ============================================================
def distill_one_round(
    teacher,
    student_ddp,
    train_loader,
    train_sampler,
    teacher_timesteps,
    student_timesteps,
    alphas_cumprod,
    config,
    round_idx,
    device,
    logger,
    rank,
    world_size
):
    distill_cfg = config["distillation"]
    epochs = get_epochs_for_round(distill_cfg["epochs_per_round"], round_idx)
    lr = distill_cfg["learning_rate"]
    log_interval = distill_cfg.get("log_interval", 50)
    save_interval = distill_cfg.get("save_interval", 5)
    save_dir = distill_cfg["save_dir"]
    grad_clip = distill_cfg.get("gradient_clip", 1.0)
    rollout_back_steps = distill_cfg.get("rollout", {}).get("back_steps", 1)
    debug_shapes = distill_cfg.get("debug_shapes", False)
    use_fp16 = distill_cfg.get("use_fp16", False)

    scaler = GradScaler("cuda", enabled=use_fp16)

    pairs = build_timestep_quads(teacher_timesteps, student_timesteps)
    if len(pairs) == 0:
        logger.warning("No timestep quads found for distillation! Skipping round.")
        return student_ddp

    logger.info(f"Start distill_one_round: round={round_idx}, teacher_steps={len(teacher_timesteps)}, student_steps={len(student_timesteps)}")
    logger.info(f"Round {round_idx}: epochs={epochs}, use_fp16={use_fp16}, num_quads={len(pairs)}")

    pair_sampler = PairPrioritySampler(
        pairs=pairs,
        alpha=distill_cfg.get("priority_sampling", {}).get("alpha", 0.6),
        eps=distill_cfg.get("priority_sampling", {}).get("eps", 1e-3),
        ema_decay=distill_cfg.get("priority_sampling", {}).get("ema_decay", 0.9),
        warmup_uniform_epochs=distill_cfg.get("priority_sampling", {}).get("warmup_uniform_epochs", 1),
    )

    optimizer = torch.optim.AdamW(
        student_ddp.parameters(),
        lr=lr,
        weight_decay=distill_cfg.get("weight_decay", 0.0)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(len(train_loader), 1), eta_min=lr * 0.01
    )

    student_ddp.train()
    teacher.eval()

    for epoch in range(epochs):
        train_sampler.set_epoch(round_idx * 1000 + epoch)
        source_probs = get_source_mix_probs(round_idx, epoch, epochs, distill_cfg)

        logger.info(
            f"Round {round_idx} Epoch {epoch+1}/{epochs} source mix: "
            f"forward={source_probs['forward']:.2f}, "
            f"teacher={source_probs['teacher']:.2f}, "
            f"student={source_probs['student']:.2f}"
        )

        epoch_loss = 0.0
        epoch_loss_x = 0.0
        epoch_loss_x0 = 0.0
        epoch_loss_grad = 0.0
        epoch_loss_spec = 0.0
        num_batches = 0
        source_counter = {"forward": 0, "teacher": 0, "student": 0, "forward_fallback": 0}

        if rank == 0:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Round {round_idx} Epoch {epoch+1}/{epochs}",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            pbar = enumerate(train_loader)

        try:
            for batch_idx, data_dict in pbar:
                if batch_idx == 0:
                    logger.info(f"Round {round_idx} Epoch {epoch+1}: first batch start")

                x_0 = data_dict["image"].to(device, non_blocking=True)
                cond = data_dict["condition"].to(device, non_blocking=True)

                if x_0.dim() == 3:
                    x_0 = x_0.unsqueeze(0)
                if cond.dim() == 3:
                    cond = cond.unsqueeze(0)

                pair_idx, (t_start, t_mid1, t_mid2, t_mid3, t_end) = pair_sampler.sample(epoch)
                source = sample_state_source(source_probs)

                if source == "forward":
                    x_t = sample_forward_noised_state(x_0, t_start, alphas_cumprod)
                    source_name = "forward"
                elif source == "teacher":
                    x_t, source_name = sample_teacher_rollout_state(
                        teacher, x_0, cond, t_start, student_timesteps, alphas_cumprod,
                        rollout_back_steps=rollout_back_steps, use_fp16=use_fp16
                    )
                else:
                    with torch.no_grad():
                        x_t, source_name = sample_student_rollout_state(
                            student_ddp, x_0, cond, t_start, student_timesteps, alphas_cumprod,
                            rollout_back_steps=rollout_back_steps, use_fp16=use_fp16
                        )

                source_counter[source_name] = source_counter.get(source_name, 0) + 1

                if debug_shapes and batch_idx == 0:
                    logger.info(
                        f"[DEBUG SHAPES] x_0={tuple(x_0.shape)}, cond={tuple(cond.shape)}, "
                        f"x_t={tuple(x_t.shape)}, quad=({t_start},{t_mid1},{t_mid2},{t_mid3},{t_end}), source={source_name}"
                    )

                with torch.no_grad():
                    x_target, x0_target = teacher_four_steps(
                        teacher, x_t, t_start, t_mid1, t_mid2, t_mid3, t_end, cond, alphas_cumprod, use_fp16=use_fp16
                    )

                with autocast("cuda", enabled=use_fp16):
                    eps_s = model_forward(student_ddp, x_t, t_start, cond)
                    x_pred, x0_pred = ddim_step(x_t, eps_s, alphas_cumprod, t_start, t_end)

                    loss, loss_dict = compute_distill_loss(
                        x_pred, x_target, x0_pred, x0_target, distill_cfg
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_ddp.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                pair_sampler.update(pair_idx, float(loss.detach().item()))

                reduced = torch.tensor(
                    [
                        loss.detach().item(),
                        loss_dict["loss_x"].item(),
                        loss_dict["loss_x0"].item(),
                        loss_dict["loss_grad"].item(),
                        loss_dict["loss_spec"].item(),
                    ],
                    device=device,
                    dtype=torch.float32
                )
                reduced = reduce_mean(reduced, world_size)

                epoch_loss += reduced[0].item()
                epoch_loss_x += reduced[1].item()
                epoch_loss_x0 += reduced[2].item()
                epoch_loss_grad += reduced[3].item()
                epoch_loss_spec += reduced[4].item()
                num_batches += 1

                if rank == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_loss_x = epoch_loss_x / num_batches
                    avg_loss_x0 = epoch_loss_x0 / num_batches
                    avg_loss_grad = epoch_loss_grad / num_batches
                    avg_loss_spec = epoch_loss_spec / num_batches

                    if hasattr(pbar, "set_postfix"):
                        pbar.set_postfix(
                            loss=f"{reduced[0].item():.4f}",
                            avg=f"{avg_loss:.4f}",
                            x=f"{avg_loss_x:.4f}",
                            x0=f"{avg_loss_x0:.4f}",
                            grad=f"{avg_loss_grad:.4f}",
                            spec=f"{avg_loss_spec:.4f}",
                            src=source_name,
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                            fp16=str(use_fp16)
                        )

                    if (batch_idx + 1) % log_interval == 0:
                        stats = pair_sampler.stats()
                        logger.info(
                            f"Round {round_idx} | Epoch {epoch+1}/{epochs} | Batch {batch_idx+1} | "
                            f"Loss: {reduced[0].item():.6f} | Avg: {avg_loss:.6f} | "
                            f"x: {avg_loss_x:.6f} | x0: {avg_loss_x0:.6f} | "
                            f"grad: {avg_loss_grad:.6f} | spec: {avg_loss_spec:.6f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                            f"Quad: ({t_start},{t_mid1},{t_mid2},{t_mid3},{t_end}) | Source: {source_name} | "
                            f"PairEMA[min/mean/max]=({stats['min']:.4f}/{stats['mean']:.4f}/{stats['max']:.4f})"
                        )
        finally:
            if rank == 0 and hasattr(pbar, "close"):
                pbar.close()

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_epoch_loss_x = epoch_loss_x / max(num_batches, 1)
        avg_epoch_loss_x0 = epoch_loss_x0 / max(num_batches, 1)
        avg_epoch_loss_grad = epoch_loss_grad / max(num_batches, 1)
        avg_epoch_loss_spec = epoch_loss_spec / max(num_batches, 1)

        logger.info(
            f"Round {round_idx} | Epoch {epoch+1}/{epochs} done | "
            f"Avg Loss: {avg_epoch_loss:.6f} | x: {avg_epoch_loss_x:.6f} | x0: {avg_epoch_loss_x0:.6f} | "
            f"grad: {avg_epoch_loss_grad:.6f} | spec: {avg_epoch_loss_spec:.6f} | "
            f"LocalSources: {source_counter}"
        )

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(save_dir, f"student_round{round_idx}_epoch{epoch+1}.pt")
            save_student_checkpoint(
                student_ddp,
                ckpt_path,
                {
                    "round": round_idx,
                    "epoch": epoch + 1,
                    "num_steps": len(student_timesteps),
                    "student_timesteps": student_timesteps.tolist(),
                    "loss": avg_epoch_loss,
                    "priority_ema_losses": pair_sampler.ema_losses.tolist(),
                    "use_fp16": use_fp16,
                },
                rank,
                scaler=scaler
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    student_ddp.eval()
    return student_ddp


# ============================================================
# Worker
# ============================================================
def main_worker(rank, world_size, args, config):
    os.environ["MASTER_ADDR"] = config["distillation"].get("master_addr", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(config["distillation"].get("master_port", 29500))

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    distill_cfg = config["distillation"]
    save_dir = distill_cfg["save_dir"]
    logger = setup_logger(save_dir, rank)

    setup_seed(distill_cfg.get("seed", 1234), rank)
    logger.info(f"Spawn DDP initialized with world_size={world_size}, device={device}")

    diffusion_config = load_yaml(config["teacher"]["diffusion_config"])
    noise_schedule = diffusion_config.get("noise_schedule", "linear")
    total_steps = diffusion_config.get("steps", 1000)
    alphas_cumprod = build_alpha_schedule(noise_schedule, total_steps)
    logger.info(f"Built alpha schedule: {noise_schedule}, {total_steps} steps")

    data_cfg = dict(config.get('data', {}))
    data_cfg.setdefault('name', 'zarr_climate')
    data_cfg.setdefault('flag', 'train')
    dataset = get_dataset(**data_cfg)

    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    num_workers = distill_cfg.get("num_workers", 0)
    persistent_workers = distill_cfg.get("persistent_workers", False) and (num_workers > 0)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=distill_cfg.get("batch_size", 1),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
    )

    logger.info(
        f"Dataset: {len(dataset)} samples, "
        f"per_gpu_batch_size={distill_cfg.get('batch_size', 1)}, "
        f"num_workers={num_workers}, "
        f"persistent_workers={persistent_workers}, "
        f"use_fp16={distill_cfg.get('use_fp16', False)}"
    )

    num_rounds = distill_cfg["num_rounds"]
    initial_steps = distill_cfg["initial_steps"]
    auto_resume = distill_cfg.get("auto_resume", True)
    latest_completed_round = -1

    if auto_resume:
        latest_completed_round = detect_latest_completed_round(save_dir, num_rounds)

    if latest_completed_round >= 0:
        start_round = latest_completed_round + 1
        current_teacher_ckpt = os.path.join(save_dir, f"student_round{latest_completed_round}_final.pt")
        current_steps = get_steps_after_rounds(initial_steps, start_round)
        logger.info(
            f"[AUTO RESUME] Detected completed rounds up to round {latest_completed_round}. "
            f"Resuming from round {start_round}, teacher_ckpt={current_teacher_ckpt}, current_steps={current_steps}"
        )
    else:
        start_round = 0
        current_teacher_ckpt = config["teacher"]["checkpoint_path"]
        current_steps = initial_steps
        logger.info("[AUTO RESUME] No completed round checkpoint found. Starting from round 0.")

    if start_round >= num_rounds:
        logger.info(f"[AUTO RESUME] All rounds already completed (start_round={start_round}, num_rounds={num_rounds}). Exiting.")
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
        return

    try:
        for round_idx in range(start_round, num_rounds):
            logger.info(f"[ENTER ROUND] rank={rank}, round={round_idx}, current_steps={current_steps}")

            target_steps = current_steps // 4
            if target_steps < 2:
                logger.warning(f"Cannot distill further: target_steps={target_steps} < 2. Stopping.")
                break

            teacher_timesteps = build_ddim_timesteps(total_steps, current_steps)
            student_timesteps = build_ddim_timesteps(total_steps, target_steps)

            logger.info(
                f"Round {round_idx}: teacher_steps={len(teacher_timesteps)}, student_steps={len(student_timesteps)}"
            )

            current_teacher = load_teacher_model(
                config["teacher"]["model_config"],
                current_teacher_ckpt,
                device,
                logger,
                rank
            )
            current_teacher.eval()

            student = copy.deepcopy(unwrap_model(current_teacher))
            student.to(device)

            student_ddp = DDP(
                student,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False
            )

            student_ddp = distill_one_round(
                teacher=current_teacher,
                student_ddp=student_ddp,
                train_loader=train_loader,
                train_sampler=train_sampler,
                teacher_timesteps=teacher_timesteps,
                student_timesteps=student_timesteps,
                alphas_cumprod=alphas_cumprod,
                config=config,
                round_idx=round_idx,
                device=device,
                logger=logger,
                rank=rank,
                world_size=world_size,
            )

            final_ckpt = os.path.join(save_dir, f"student_round{round_idx}_final.pt")
            save_student_checkpoint(
                student_ddp,
                final_ckpt,
                {
                    "round": round_idx,
                    "num_steps": target_steps,
                    "student_timesteps": student_timesteps.tolist(),
                    "use_fp16": distill_cfg.get("use_fp16", False),
                },
                rank
            )

            logger.info(f"Round {round_idx} final checkpoint prepared: {final_ckpt}")
            dist.barrier()
            logger.info(f"Round {round_idx}: passed save barrier")

            del student_ddp
            del student
            del current_teacher
            torch.cuda.empty_cache()

            current_teacher_ckpt = final_ckpt
            current_steps = target_steps

            dist.barrier()
            logger.info(f"Round {round_idx}: passed cleanup barrier")

        logger.info(f"Distillation complete! Final model steps={current_steps}")

    finally:
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Improved Progressive Distillation DDP Spawn + AMP + auto resume")
    parser.add_argument("--config", type=str, required=True, help="Path to distill_config.yaml")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use. Default: all visible GPUs")
    args = parser.parse_args()

    config = load_yaml(args.config)

    world_size = args.gpus if args.gpus is not None else torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found.")
    if world_size > torch.cuda.device_count():
        raise ValueError("--gpus cannot exceed the number of visible CUDA devices.")

    if "master_port" not in config["distillation"]:
        config["distillation"]["master_port"] = find_free_port()

    mp.spawn(
        main_worker,
        nprocs=world_size,
        args=(world_size, args, config),
        join=True
    )


if __name__ == "__main__":
    main()
