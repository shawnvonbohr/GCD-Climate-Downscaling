"""Run distilled conditional diffusion downscaling with CG data consistency.

The script starts one worker per visible CUDA device, shards MIROC6 time steps
between them, and writes physical-unit reconstructions to one Zarr store.
"""

import os
import sys
import argparse
import time
import importlib
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

# Add the project root for direct script execution.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from script.condition_methods_distill import get_conditioning_method
from script.model import DiffusersUNetWrapper, load_yaml
from guided_diffusion.measurements import get_noise, get_operator
from util.logger import get_logger
from script.distilled_sampler import create_distilled_sampler, get_named_beta_schedule

# ============================================================
# Optional diagnostic plots: restore physical units before rendering.
# ============================================================
def save_color_images_per_channel(data_tensor, out_path, prefix, fname_base,
                                  channel_mean, channel_std, var_names,
                                  fixed_vmin=None, fixed_vmax=None):
    import matplotlib.pyplot as plt

    if data_tensor.ndim == 4:
        data_tensor = data_tensor[0]
    elif data_tensor.ndim != 3:
        raise ValueError(f"Expected 3D (C,H,W) or 4D (B,C,H,W) data, got {data_tensor.ndim}D.")

    C = data_tensor.shape[0]
    if hasattr(data_tensor, 'detach'):
        data_np_norm = data_tensor.detach().cpu().float().numpy()
    else:
        data_np_norm = np.asarray(data_tensor, dtype=np.float64)

    if C != len(var_names):
        var_names = [f'channel_{i}' for i in range(C)]

    save_dir = os.path.join(out_path, prefix)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(C):
        var_name = var_names[i]
        channel_data_norm = data_np_norm[i]

        # Restore physical units.
        channel_data_raw = channel_data_norm * channel_std[i] + channel_mean[i]

        fig, ax = plt.subplots(figsize=(6, 6))

        if fixed_vmin is not None and fixed_vmax is not None:
            v_min = fixed_vmin[i] if isinstance(fixed_vmin, np.ndarray) else fixed_vmin
            v_max = fixed_vmax[i] if isinstance(fixed_vmax, np.ndarray) else fixed_vmax
        else:
            v_min = channel_data_raw.min()
            v_max = channel_data_raw.max()

        im = ax.imshow(channel_data_raw, cmap='rainbow', vmin=v_min, vmax=v_max)
        ax.set_title(f'{prefix.capitalize()} - {var_name}', fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

        save_path = os.path.join(save_dir, f"{fname_base}_{prefix}_{var_name}.png")
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


# ============================================================
# DDP Setup / Cleanup
# ============================================================
def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12356')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dataloader_module(dataloader_name):
    module_path = f"data.{dataloader_name}"
    return importlib.import_module(module_path)


# ============================================================
# Load the distilled model.
# ============================================================
def load_distilled_model(model_config, distilled_ckpt_path, device, logger):
    model = DiffusersUNetWrapper(model_config, device)

    logger.info(f"Loading distilled checkpoint: {distilled_ckpt_path}")
    ckpt = torch.load(distilled_ckpt_path, map_location='cpu', weights_only=False)

    if not isinstance(ckpt, dict) or not ckpt.get('state_dict'):
        raise ValueError(
            "The checkpoint must be a dictionary containing a 'state_dict' entry."
        )
    if 'student_timesteps' not in ckpt or 'num_steps' not in ckpt:
        raise ValueError(
            "The checkpoint must contain 'student_timesteps' and 'num_steps' metadata."
        )

    model.unet.load_state_dict(ckpt['state_dict'], strict=True)
    logger.info("Distilled model weights loaded.")

    for module in model.modules():
        if isinstance(module, torch.nn.GroupNorm) and isinstance(module.eps, str):
            module.eps = float(module.eps)

    try:
        model.unet.enable_attention_slicing(slice_size="auto")
        logger.info("Enabled Attention Slicing.")
    except Exception:
        pass

    model.eval()

    student_timesteps = ckpt.get('student_timesteps', None)
    num_steps = ckpt.get('num_steps', None)
    logger.info(f"Distilled model: {num_steps} steps, timesteps={student_timesteps}")

    return model, student_timesteps, num_steps


# ============================================================
# Main Worker
# ============================================================
def main_worker(rank, world_size, args, model_config, diffusion_config, task_config):
    setup(rank, world_size)

    logger = get_logger()
    if rank != 0:
        logger.setLevel(50)

    start_time_total = time.time()
    device = torch.device(f"cuda:{rank}")
    logger.info(f"Device: cuda:{rank}")

    # Load the distilled model.
    model, student_timesteps, num_steps = load_distilled_model(
        model_config, args.distilled_ckpt, device, logger
    )
    if student_timesteps is None:
        raise ValueError("Checkpoint does not contain student_timesteps!")

    # Build the distilled sampler.
    noise_schedule = diffusion_config.get('noise_schedule', 'linear')
    total_steps = diffusion_config.get('steps', 1000)
    betas = get_named_beta_schedule(noise_schedule, total_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    sampler = create_distilled_sampler(
        alphas_cumprod=alphas_cumprod,
        student_timesteps=student_timesteps,
        rescale_timesteps=diffusion_config.get('rescale_timesteps', False),
    )
    logger.info(f"DistilledDDIM sampler: {sampler.num_timesteps} steps")

    # Build the observation operator and conditioning method.
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operator: {measure_config['operator']['name']} | Noise: {measure_config['noise']['name']}")

    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning: {cond_config['method']} | scale: {cond_config['params'].get('scale', 'N/A')}")

    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Create the output directory on rank zero.
    out_path = task_config.get('output', {}).get('save_path',
                os.path.join(args.save_dir, measure_config['operator']['name']))
    output_config = task_config.get('output', {})
    save_images = output_config.get('save_images', False)

    if rank == 0:
        os.makedirs(out_path, exist_ok=True)
        if save_images:
            for img_dir in ['input', 'recon', 'label']:
                os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
        logger.info(f"Save path: {out_path}")

    # Dynamically import the configured data loader.
    data_config = task_config['data']
    dataloader_name = data_config.get('dataloader', 'dataloader_accelerate_miroc6')
    if rank == 0:
        logger.info(f"Dataloader module: {dataloader_name}")

    dataloader_module = get_dataloader_module(dataloader_name)
    get_dataset = dataloader_module.get_dataset
    get_dataloader = dataloader_module.get_dataloader

    dataset = get_dataset(**data_config)
    total_samples = len(dataset)
    if rank == 0:
        logger.info(f"Dataset total samples: {total_samples}")

    dist_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = get_dataloader(
        dataset,
        batch_size=data_config.get('batch_size', 1),
        num_workers=data_config.get('num_workers', 0),
        train=False,
        sampler=dist_sampler,
    )

    # Collect normalization statistics.
    C_expected = model_config['out_channels']
    try:
        data_mean_raw = dataset.mean
        data_std_raw = dataset.std

        if isinstance(data_mean_raw, torch.Tensor):
            data_mean_full = data_mean_raw.flatten().cpu().numpy()
        else:
            data_mean_full = np.array(data_mean_raw).flatten()
        if isinstance(data_std_raw, torch.Tensor):
            data_std_full = data_std_raw.flatten().cpu().numpy()
        else:
            data_std_full = np.array(data_std_raw).flatten()

        data_var_names_used = getattr(dataset, 'var_names', [f'channel_{i}' for i in range(len(data_mean_full))])
        data_mean = data_mean_full[dataset.var_index].astype(np.float32)
        data_std = data_std_full[dataset.var_index].astype(np.float32)
        data_var_names = list(data_var_names_used)
        if len(data_mean) != C_expected or len(data_var_names) != C_expected:
            raise ValueError(
                "Dataset variables must match the model's out_channels: "
                f"got {len(data_var_names)} variables for {C_expected} output channels."
            )
    except Exception as e:
        raise RuntimeError("Could not prepare dataset normalization statistics.") from e

    if rank == 0:
        logger.info(f"Batches per GPU: {len(loader)}")

    # Initialize the Zarr output on rank zero.
    new_zarr = os.path.join(out_path, "results.zarr")
    latitude_coord = dataset.target_lat
    longitude_coord = dataset.target_lon

    written_time_keys = set()
    if rank == 0:
        import xarray as xr
        if not os.path.exists(new_zarr):
            logger.info(f"Initializing Zarr: {new_zarr}")
            z_data = (['time', 'channel', 'lat', 'lon'],
                      np.empty((0, C_expected, len(latitude_coord), len(longitude_coord)), dtype=np.float32))
            empty_dataset = xr.Dataset(
                {"data": z_data},
                coords={
                    'time': np.array([], dtype='datetime64[ns]'),
                    'channel': data_var_names,
                    'lat': latitude_coord,
                    'lon': longitude_coord,
                }
            )
            encoding = {
                "data": {'chunks': (1, 1, len(latitude_coord), len(longitude_coord))},
                'time': {'units': 'hours since 1900-01-01', 'dtype': 'int64'}
            }
            empty_dataset.to_zarr(new_zarr, mode='w', encoding=encoding)
        else:
            logger.info(f"Existing Zarr found: {new_zarr}, will append.")
            existing_times = xr.open_zarr(new_zarr)['time'].values
            written_time_keys = {
                np.datetime64(value).astype('datetime64[ns]').astype('int64')
                for value in existing_times
            }

    dist.barrier()

    # ---- tqdm (rank 0 only) ----
    if rank == 0:
        from tqdm import tqdm
        loader = tqdm(loader, desc=f"Distilled DPS ({sampler.num_timesteps} steps)")

    # ---- Autocast ----
    is_enabled = False
    dtype = torch.float32
    if device.type == 'cuda':
        try:
            props = torch.cuda.get_device_properties(device)
            if props.major >= 7:
                is_enabled = True
                dtype = torch.float16
        except Exception:
            pass
    autocast_context = torch.autocast(device_type=device.type, dtype=dtype, enabled=is_enabled)

    # ---- Inference Loop ----
    for i, data_dict in enumerate(loader):
        sample_ids = data_dict['sample_id']

        ref_img = data_dict['image'].to(device)
        fixed_condition = data_dict['condition'].to(device)

        if ref_img.dim() == 3:
            ref_img = ref_img.unsqueeze(0)
        if fixed_condition.dim() == 3:
            fixed_condition = fixed_condition.unsqueeze(0)

        if ref_img.shape[1] != C_expected:
            raise ValueError(f"Expected {C_expected} data channels, got {ref_img.shape[1]}.")
        if tuple(ref_img.shape[-2:]) != tuple(model_config['sample_size']):
            raise ValueError(
                f"Dataset resolution {tuple(ref_img.shape[-2:])} does not match "
                f"model sample_size {tuple(model_config['sample_size'])}."
            )

        # Construct the noisy observation.
        y_n = noiser(ref_img)

        sample_height, sample_width = model_config['sample_size']
        x_start = torch.randn(
            ref_img.shape[0], ref_img.shape[1], sample_height, sample_width, device=device
        )

        if rank == 0:
            logger.info(f"Batch {i}: Starting distilled DPS ({sampler.num_timesteps} steps)...")
        t0 = time.time()

        with autocast_context:
            sample = sample_fn(
                x_start=x_start,
                measurement=y_n,
                fixed_condition=fixed_condition,
                record=False,
                save_root=out_path,
            )

        if rank == 0:
            logger.info(f"Batch {i}: Done in {time.time()-t0:.2f}s")

        if save_images:
            ref_img_cpu = ref_img.detach().cpu().float()
            sample_cpu = sample.detach().cpu().float()
            y_n_cpu = y_n.detach().cpu().float()
            ref_img_np = ref_img_cpu.numpy()

            for b in range(ref_img_np.shape[0]):
                current_id = sample_ids[b]
                clean_id = current_id.replace(':', '_')
                fname_base = f"rank{rank}_{clean_id}"
                unified_v_min = np.empty(C_expected, dtype=np.float32)
                unified_v_max = np.empty(C_expected, dtype=np.float32)
                for i_channel in range(C_expected):
                    data_raw = ref_img_np[b, i_channel] * data_std[i_channel] + data_mean[i_channel]
                    unified_v_min[i_channel] = data_raw.min()
                    unified_v_max[i_channel] = data_raw.max()

                for image, prefix in ((y_n_cpu[b], 'input'), (ref_img_cpu[b], 'label'), (sample_cpu[b], 'recon')):
                    try:
                        save_color_images_per_channel(
                            image, out_path, prefix, fname_base, data_mean, data_std,
                            data_var_names, fixed_vmin=unified_v_min,
                            fixed_vmax=unified_v_max,
                        )
                    except Exception:
                        logger.exception(f"Rank {rank}: Failed to save {prefix}: {fname_base}")

        # Gather samples and append them to Zarr on rank zero.
        gathered_samples_list = [torch.empty_like(sample) for _ in range(world_size)] if rank == 0 else []
        torch.distributed.gather(sample.float().contiguous(), gather_list=gathered_samples_list, dst=0)

        gathered_ids_list = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(sample_ids, gathered_ids_list, dst=0)

        if rank == 0:
            import xarray as xr
            combined_data = torch.cat(gathered_samples_list, dim=0).cpu().numpy()
            combined_data = (
                combined_data * data_std[None, :, None, None]
                + data_mean[None, :, None, None]
            ).astype(np.float32, copy=False)
            combined_ids_flat = [ts for sublist in gathered_ids_list for ts in sublist]
            sample_ids_final = np.array(
                [np.datetime64(x).astype('datetime64[ns]') for x in combined_ids_flat]
            )
            candidate_time_keys = set(written_time_keys)
            new_mask = []
            for timestamp in sample_ids_final:
                time_key = timestamp.astype('int64')
                is_new = time_key not in candidate_time_keys
                new_mask.append(is_new)
                if is_new:
                    candidate_time_keys.add(time_key)
            new_mask = np.asarray(new_mask, dtype=bool)
            combined_data = combined_data[new_mask]
            sample_ids_final = sample_ids_final[new_mask]

            if not len(sample_ids_final):
                logger.info(f"Batch {i}: all time steps were already written; skipping.")
                dist.barrier()
                continue

            sub_dataset = xr.Dataset(
                {"data": (['time', 'channel', 'lat', 'lon'], combined_data)},
                coords={
                    'time': sample_ids_final,
                    'channel': data_var_names,
                    'lat': latitude_coord,
                    'lon': longitude_coord,
                }
            )

            sub_dataset['data'].attrs['normalization'] = 'physical units; mean/std restored before writing'
            try:
                sub_dataset.to_zarr(new_zarr, mode='a', append_dim='time')
                written_time_keys.update(timestamp.astype('int64') for timestamp in sample_ids_final)
                logger.info(f"Batch {i}: Appended {len(sample_ids_final)} steps to Zarr.")
            except Exception as e:
                logger.exception(f"Failed to append to Zarr: {e}")

        dist.barrier()

    total_time = time.time() - start_time_total
    logger.info(f"Rank {rank}: Total inference time: {total_time:.2f}s")
    cleanup()


# ============================================================
# Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Distilled DPS Inference (DDP)")
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--diffusion_config', type=str, required=True)
    parser.add_argument('--task_config', type=str, required=True)
    parser.add_argument('--distilled_ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='results/distilled_dps')
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPUs.")
    if world_size == 0:
        raise RuntimeError("This distilled-CG inference entry point requires at least one CUDA GPU.")

    if world_size > 1:
        mp.spawn(main_worker, nprocs=world_size,
                 args=(world_size, args, model_config, diffusion_config, task_config))
    else:
        main_worker(0, 1, args, model_config, diffusion_config, task_config)


if __name__ == '__main__':
    main()
