"""Train the ERA5 conditional diffusion prior used as the distillation teacher."""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import socket
import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.era5_prior_dataset import ERA5PriorDataset
from script.model import DiffusersUNetWrapper, load_yaml


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def set_seed(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model: Union[DDP, DiffusersUNetWrapper]) -> DiffusersUNetWrapper:
    return model.module if isinstance(model, DDP) else model


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_noise_scheduler(diffusion_config: dict) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=int(diffusion_config.get("steps", 1000)),
        beta_schedule=diffusion_config.get("noise_schedule", "linear"),
        clip_sample=bool(diffusion_config.get("clip_denoised", False)),
        prediction_type="epsilon",
    )


def save_checkpoint(
    model: Union[DDP, DiffusersUNetWrapper],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    checkpoint_path: Path,
    epoch: int,
    global_step: int,
    model_config: dict,
    diffusion_config: dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped_model = unwrap_model(model)
    torch.save(
        {
            "state_dict": unwrapped_model.unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "num_steps": int(diffusion_config.get("steps", 1000)),
            "model_config": model_config,
            "diffusion_config": diffusion_config,
        },
        checkpoint_path,
    )


def get_autocast_settings(mixed_precision: str) -> Tuple[bool, torch.dtype]:
    if mixed_precision == "no":
        return False, torch.float32
    if mixed_precision == "fp16":
        return True, torch.float16
    if mixed_precision == "bf16":
        return True, torch.bfloat16
    raise ValueError("mixed_precision must be one of: no, fp16, bf16.")


def main_worker(rank: int, world_size: int, config: dict) -> None:
    training_config = config["training"]
    os.environ["MASTER_ADDR"] = training_config.get("master_addr", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(training_config["master_port"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        set_seed(int(training_config.get("seed", 1234)), rank)

        model_config = load_yaml(config["model"]["config"])
        diffusion_config = load_yaml(config["diffusion"]["config"])
        dataset = ERA5PriorDataset(**config["data"])
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        num_workers = int(training_config.get("num_workers", 0))
        dataloader = DataLoader(
            dataset,
            batch_size=int(training_config.get("batch_size", 1)),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0 and bool(training_config.get("persistent_workers", False)),
        )
        if len(dataloader) == 0:
            raise ValueError(
                "No training batches were created. Reduce batch_size or use a larger time range."
            )

        model = DiffusersUNetWrapper(model_config, device)
        model = DDP(model, device_ids=[rank], output_device=rank)
        noise_scheduler = build_noise_scheduler(diffusion_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config.get("learning_rate", 1.0e-4)),
            weight_decay=float(training_config.get("weight_decay", 0.0)),
        )
        num_epochs = int(training_config["num_epochs"])
        lr_scheduler = build_lr_scheduler(
            optimizer,
            total_steps=num_epochs * len(dataloader),
            warmup_steps=int(training_config.get("warmup_steps", 0)),
        )

        mixed_precision = training_config.get("mixed_precision", "fp16")
        use_autocast, autocast_dtype = get_autocast_settings(mixed_precision)
        scaler = GradScaler("cuda", enabled=mixed_precision == "fp16")
        gradient_clip = float(training_config.get("gradient_clip", 1.0))
        log_every_steps = int(training_config.get("log_every_steps", 50))
        save_every_epochs = int(training_config.get("save_every_epochs", 10))
        checkpoint_path = Path(training_config["checkpoint_path"])
        global_step = 0

        if rank == 0:
            print(
                "Training ERA5 diffusion prior: "
                f"{len(dataset)} samples, {world_size} GPU(s), "
                f"per-GPU batch size {training_config.get('batch_size', 1)}."
            )

        for epoch in range(1, num_epochs + 1):
            sampler.set_epoch(epoch)
            model.train()
            loss_sum = 0.0
            batch_count = 0
            progress = tqdm(
                dataloader,
                desc=f"Prior epoch {epoch}/{num_epochs}",
                disable=rank != 0,
                dynamic_ncols=True,
            )
            for batch in progress:
                clean_images = batch["image"].to(device, non_blocking=True)
                conditions = batch["condition"].to(device, non_blocking=True)
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (clean_images.shape[0],),
                    device=device,
                    dtype=torch.long,
                )
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=use_autocast, dtype=autocast_dtype):
                    noise_prediction = model(noisy_images, timesteps, conditions)
                    loss = F.mse_loss(noise_prediction.float(), noise.float())
                scaler.scale(loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                batch_loss = float(loss.detach())
                loss_sum += batch_loss
                batch_count += 1
                global_step += 1
                if rank == 0:
                    progress.set_postfix(
                        loss=f"{batch_loss:.5f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    )
                    if global_step % log_every_steps == 0:
                        print(
                            f"Epoch {epoch}/{num_epochs}, step {global_step}: "
                            f"loss={batch_loss:.6f}"
                        )

            epoch_stats = torch.tensor([loss_sum, batch_count], device=device, dtype=torch.float64)
            dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
            mean_loss = (epoch_stats[0] / epoch_stats[1].clamp_min(1)).item()
            if rank == 0:
                print(f"Epoch {epoch}/{num_epochs} complete: mean loss={mean_loss:.6f}")
                if epoch % save_every_epochs == 0 or epoch == num_epochs:
                    snapshot_path = checkpoint_path.with_name(
                        f"{checkpoint_path.stem}_epoch{epoch}{checkpoint_path.suffix}"
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        scaler,
                        snapshot_path,
                        epoch,
                        global_step,
                        model_config,
                        diffusion_config,
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        scaler,
                        checkpoint_path,
                        epoch,
                        global_step,
                        model_config,
                        diffusion_config,
                    )
                    print(f"Saved teacher checkpoint: {checkpoint_path}")
            dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ERA5 diffusion prior used as the GCD teacher model."
    )
    parser.add_argument("--config", required=True, help="Path to prior-training YAML config.")
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of visible GPUs to use. Defaults to all visible GPUs.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prior training.")
    config = copy.deepcopy(load_yaml(args.config))
    if "training" not in config:
        raise KeyError("The prior-training config must contain a 'training' section.")
    world_size = args.gpus if args.gpus is not None else torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found.")
    if world_size > torch.cuda.device_count():
        raise ValueError("--gpus cannot exceed the number of visible CUDA devices.")
    config["training"].setdefault("master_port", find_free_port())

    mp.spawn(main_worker, nprocs=world_size, args=(world_size, config), join=True)


if __name__ == "__main__":
    main()
