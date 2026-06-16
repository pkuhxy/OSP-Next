import os
import sys
import math
import yaml
import time
import json
import random
import tempfile
import faulthandler
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent import futures
from argparse import ArgumentParser

import wandb
import imageio

from ospnext.utils.utils import check_and_import_npu, is_npu_available
import torch
check_and_import_npu()

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from ospnext.utils.log_utils import get_logger, log_on_main_process, verify_min_gpu_count
from ospnext.utils.random_utils import set_seed
from ospnext.distributed.utils import (
    setup_distributed_env,
    cleanup_distributed_env,
    set_modules_to_forward_prefetch,
    set_modules_to_backward_prefetch,
    gather_data_from_all_ranks,
)
from ospnext.distributed.fsdp2_wrapper import FSDP2_mix_wrapper
from ospnext.distributed.fsdp_ema import FSDPEMAModel as EMAModel
from ospnext.distributed.sp_state import sp_state

from ospnext.modules import (
    WanVAE,
    T5EncoderModel,
    models,
    models_main_block,
    models_blocks_to_float,
    models_blocks_to_output_float,
)
from ospnext.schedulers import schedulers

from ospnext.utils.constant import PROMPT, PROMPT_IDS, PROMPT_MASK, VIDEO
from ospnext.utils.utils import str_to_precision, params_nums_to_str, get_memory_allocated
from ospnext.utils.clip_grads import AdaptiveGradClipper
from ospnext.data.utils.wan_utils import WanTextProcessor, WanVideoProcessor
from transformers import AutoTokenizer

from peft import LoraConfig, get_peft_model, PeftModel


def get_ddp_rank_and_fsdp_local_rank(rank, fsdp_size, world_size):
    ddp_size = max(1, world_size // fsdp_size)
    ddp_rank = rank // fsdp_size
    fsdp_local_rank = rank % fsdp_size
    return ddp_rank, fsdp_local_rank, ddp_size


def _as_float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value, got {value!r}") from exc

def sde_step_with_logprob(
    sigmas_schedule,
    model_output,
    timestep_index,
    sample,
    num_inference_steps,
    prev_sample=None,
    generator=None,
    determistic=False,
    return_dt_and_std_dev_t=False,
    sp_group=None,
):
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    sigma = sigmas_schedule[timestep_index]
    sigma_prev = sigmas_schedule[timestep_index + 1]
    sigma_max = sigmas_schedule[0].item()
    sigma_min = sigmas_schedule[-1].item()

    dt = sigma_prev - sigma

    sigma_b = sigma.view(1, 1, 1, 1, 1) if sigma.dim() == 0 else sigma.view(-1, 1, 1, 1, 1)
    dt_b = dt.view(1, 1, 1, 1, 1) if dt.dim() == 0 else dt.view(-1, 1, 1, 1, 1)

    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma_b
    prev_sample_mean = (
        sample * (1 + std_dev_t ** 2 / (2 * sigma_b) * dt_b)
        + model_output * (1 + std_dev_t ** 2 * (1 - sigma_b) / (2 * sigma_b)) * dt_b
    )

    if prev_sample is not None and generator is not None:
        raise ValueError("Cannot pass both generator and prev_sample.")

    if prev_sample is None:
        if timestep_index < num_inference_steps - 1:
            variance_noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            if sp_group is not None:
                torch.distributed.broadcast(variance_noise, src=dist.get_global_rank(sp_group, 0), group=sp_group)
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt_b) * variance_noise
        else:
            prev_sample = prev_sample_mean

    if determistic:
        prev_sample = sample + dt_b * model_output

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt_b)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt_b))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_dt_and_std_dev_t:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1 * dt_b)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1 * dt_b)


@torch.no_grad()
def osp_sample_with_logprob(
    model,
    scheduler,
    vae,
    latent_shape,
    text_embeddings,
    device,
    weight_dtype,
    num_inference_steps=50,
    guidance_scale=5.0,
    negative_text_embeddings=None,
    start_frame_latents=None,
    determistic=False,
    kl_reward=0.0,
    sp_group=None,
    sde_steps=None,
):
    if sde_steps is None:
        sde_steps = num_inference_steps
    B, C, T, H, W = latent_shape
    do_cfg = guidance_scale > 1.0

    latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

    if sp_group is not None:
        torch.distributed.broadcast(latents, src=dist.get_global_rank(sp_group, 0), group=sp_group)

    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    timesteps = sigmas * 1000.0

    all_latents = [latents]  # 仅记录 SDE 步的 latent（用于训练）
    all_log_probs = []
    all_kl = []

    for i in range(num_inference_steps):
        torch.cuda.synchronize()

        is_sde_step = (i < sde_steps)

        latents_input = latents.to(weight_dtype)
        t = timesteps[i]
        t_batch = t.expand(B).to(device)

        with torch.autocast("cuda", dtype=weight_dtype):
            noise_pred = model(
                latents_input,
                t_batch,
                text_embeddings,
                start_frame_latents=start_frame_latents,
            )
        torch.cuda.synchronize()

        if do_cfg and negative_text_embeddings is not None:
            with torch.autocast("cuda", dtype=weight_dtype):
                noise_uncond = model(
                    latents_input,
                    t_batch,
                    negative_text_embeddings,
                    start_frame_latents=start_frame_latents,
                )
            torch.cuda.synchronize()
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            del noise_uncond

        latents_ori = latents.clone()

        if is_sde_step:
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                sigmas,
                noise_pred.float(),
                i,
                latents.float(),
                num_inference_steps,
                determistic=False,
                sp_group=sp_group,
            )
        else:
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                sigmas,
                noise_pred.float(),
                i,
                latents.float(),
                num_inference_steps,
                determistic=True,
                sp_group=sp_group,
            )
        del noise_pred, latents_input

        if is_sde_step:
            all_latents.append(latents)
            all_log_probs.append(log_prob)

        if is_sde_step and kl_reward > 0 and not determistic:
            with model.disable_adapter():
                with torch.autocast("cuda", dtype=weight_dtype):
                    ref_noise_pred = model(
                        latents_ori.to(weight_dtype),
                        t_batch,
                        text_embeddings,
                        start_frame_latents=start_frame_latents,
                    )
                torch.cuda.synchronize()
                if do_cfg and negative_text_embeddings is not None:
                    with torch.autocast("cuda", dtype=weight_dtype):
                        ref_noise_uncond = model(
                            latents_ori.to(weight_dtype),
                            t_batch,
                            negative_text_embeddings,
                            start_frame_latents=start_frame_latents,
                        )
                    torch.cuda.synchronize()
                    ref_noise_pred = ref_noise_uncond + guidance_scale * (ref_noise_pred - ref_noise_uncond)
                    del ref_noise_uncond

            _, ref_log_prob, ref_prev_latents_mean, ref_std_dev_t = sde_step_with_logprob(
                sigmas,
                ref_noise_pred.float(),
                i,
                latents_ori.float(),
                num_inference_steps,
                prev_sample=latents.float(),
                sp_group=sp_group,
            )
            del ref_noise_pred
            kl = ((prev_latents_mean - ref_prev_latents_mean) ** 2 / (2 * std_dev_t ** 2))
            kl = kl.mean(dim=tuple(range(1, kl.ndim)))
            all_kl.append(kl)
            del ref_prev_latents_mean, ref_std_dev_t
        elif is_sde_step:
            all_kl.append(torch.zeros(B, device=device))
        del latents_ori, prev_latents_mean, std_dev_t

        if (i + 1) % 5 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    all_latents_cpu = [l.cpu() for l in all_latents]
    all_log_probs_cpu = [lp.cpu() for lp in all_log_probs]
    all_kl_cpu = [k.cpu() for k in all_kl]
    del all_latents, all_log_probs, all_kl
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.no_grad():
        videos = vae.decode(latents)  # [B, C, T, H, W], range [-1, 1]
    del latents

    return videos, all_latents_cpu, all_log_probs_cpu, all_kl_cpu


@torch.no_grad()
def osp_sample_deterministic(
    model,
    scheduler,
    vae,
    latent_shape,
    text_embeddings,
    device,
    weight_dtype,
    num_inference_steps=50,
    guidance_scale=5.0,
    negative_text_embeddings=None,
    start_frame_latents=None,
):
    B, C, T, H, W = latent_shape
    do_cfg = guidance_scale > 1.0

    latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    timesteps = sigmas * 1000.0

    for i in range(num_inference_steps):
        torch.cuda.synchronize()

        latents_input = latents.to(weight_dtype)
        t = timesteps[i]
        t_batch = t.expand(B).to(device)

        with torch.autocast("cuda", dtype=weight_dtype):
            noise_pred = model(
                latents_input,
                t_batch,
                text_embeddings,
                start_frame_latents=start_frame_latents,
            )
        torch.cuda.synchronize()

        if do_cfg and negative_text_embeddings is not None:
            with torch.autocast("cuda", dtype=weight_dtype):
                noise_uncond = model(
                    latents_input,
                    t_batch,
                    negative_text_embeddings,
                    start_frame_latents=start_frame_latents,
                )
            torch.cuda.synchronize()
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            del noise_uncond

        latents, _, _, _ = sde_step_with_logprob(
            sigmas,
            noise_pred.float(),
            i,
            latents.float(),
            num_inference_steps,
            determistic=True,
            sp_group=None,
        )
        del noise_pred, latents_input

        if (i + 1) % 5 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    videos = vae.decode(latents)  # [B, C, T, H, W], range [-1, 1]
    del latents

    return videos


def compute_log_prob_for_training(
    model,
    sample,
    step_idx,
    text_embeddings,
    weight_dtype,
    sigmas_schedule,
    num_inference_steps,
    guidance_scale=1.0,
    negative_text_embeddings=None,
    start_frame_latents=None,
    sp_group=None,
):
    do_cfg = guidance_scale > 1.0
    latents_input = sample["latents"][:, step_idx].to(weight_dtype)
    t = (sigmas_schedule[step_idx] * 1000.0).expand(latents_input.shape[0]).to(latents_input.device)

    noise_pred = model(
        latents_input,
        t,
        text_embeddings,
        start_frame_latents=start_frame_latents,
    )


    if do_cfg and negative_text_embeddings is not None:
        with torch.no_grad():
            noise_uncond = model(
                latents_input,
                t,
                negative_text_embeddings,
                start_frame_latents=start_frame_latents,
            )
        noise_uncond = noise_uncond.detach()
        noise_pred = torch.lerp(noise_uncond, noise_pred, guidance_scale)

    if isinstance(noise_pred, DTensor):
        noise_pred = noise_pred.full_tensor()  # 保留完整的 autograd 链路

    prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
        sigmas_schedule,
        noise_pred.float(),
        step_idx,
        sample["latents"][:, step_idx].float(),
        num_inference_steps,
        prev_sample=sample["next_latents"][:, step_idx].float(),
        return_dt_and_std_dev_t=True,
        sp_group=sp_group,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt


def compute_transition_log_prob(
    model,
    x_t,
    x_t_minus_1,
    step_idx,
    text_embeddings,
    weight_dtype,
    sigmas_schedule,
    num_inference_steps,
    guidance_scale=1.0,
    negative_text_embeddings=None,
    start_frame_latents=None,
    sp_group=None,
):
    do_cfg = guidance_scale > 1.0
    latents_input = x_t.to(weight_dtype)
    t = (sigmas_schedule[step_idx] * 1000.0).expand(latents_input.shape[0]).to(latents_input.device)

    noise_pred = model(
        latents_input,
        t,
        text_embeddings,
        start_frame_latents=start_frame_latents,
    )

    if do_cfg and negative_text_embeddings is not None:
        with torch.no_grad():
            noise_uncond = model(
                latents_input,
                t,
                negative_text_embeddings,
                start_frame_latents=start_frame_latents,
            )
        noise_uncond = noise_uncond.detach()
        noise_pred = torch.lerp(noise_uncond, noise_pred, guidance_scale)

    if isinstance(noise_pred, DTensor):
        noise_pred = noise_pred.full_tensor()

    _, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
        sigmas_schedule,
        noise_pred.float(),
        step_idx,
        x_t.float(),
        num_inference_steps,
        prev_sample=x_t_minus_1.float(),
        return_dt_and_std_dev_t=True,
        sp_group=sp_group,
    )
    return log_prob, prev_sample_mean, std_dev_t, dt


def _step_weights_from_indices(step_indices, sigmas_schedule, num_inference_steps, mode="uniform", gamma=1.0):
    mode = (mode or "uniform").lower()
    if mode == "uniform":
        weights = torch.ones_like(step_indices, dtype=torch.float32)
    elif mode == "early":
        denom = max(num_inference_steps - 1, 1)
        progress = step_indices.float() / denom
        weights = (1.0 - progress).clamp_min(0.0).pow(gamma)
    elif mode == "late":
        denom = max(num_inference_steps - 1, 1)
        progress = step_indices.float() / denom
        weights = progress.clamp_min(0.0).pow(gamma)
    elif mode == "sigma":
        flat_indices = step_indices.long().clamp(0, len(sigmas_schedule) - 1)
        weights = sigmas_schedule[flat_indices].float().to(step_indices.device).pow(gamma)
    else:
        raise ValueError(
            "rl_config.offpolicy_guidance.step_weighting must be one of "
            f"'uniform', 'early', 'late', or 'sigma', got {mode!r}"
        )
    return weights / weights.mean().clamp_min(1e-6)


def compute_offpolicy_guidance_loss(
    model,
    trace_batch,
    text_embeddings,
    weight_dtype,
    sigmas_schedule,
    num_inference_steps,
    guidance_scale=1.0,
    negative_text_embeddings=None,
    loss_weight=0.01,
    adv_clip_max=5.0,
    ratio_clip=5.0,
    positive_only=True,
    use_teacher_logprob=True,
    surrogate="detached_logprob",
    step_weighting="uniform",
    step_weight_gamma=1.0,
    sp_group=None,
):
    trace_latents = trace_batch["trace_latents"]
    trace_next_latents = trace_batch["trace_next_latents"]
    trace_step_indices = trace_batch["trace_step_indices"].long()
    trace_advantages = trace_batch["trace_advantages"].float()

    bsz, num_steps = trace_step_indices.shape
    flat_x_t = trace_latents.reshape(bsz * num_steps, *trace_latents.shape[2:])
    flat_x_t_minus_1 = trace_next_latents.reshape(bsz * num_steps, *trace_next_latents.shape[2:])
    flat_step_indices = trace_step_indices.reshape(-1)
    flat_text_embeddings = (
        text_embeddings[:, None]
        .expand(bsz, num_steps, *text_embeddings.shape[1:])
        .reshape(bsz * num_steps, *text_embeddings.shape[1:])
    )

    valid = (flat_step_indices >= 0) & (flat_step_indices < num_inference_steps)
    if not valid.any():
        zero = flat_x_t.sum() * 0.0
        return loss_weight * zero, {
            "offpolicy/loss_raw": 0.0,
            "offpolicy/loss_weighted": 0.0,
            "offpolicy/active_ratio": 0.0,
            "offpolicy/ratio_mean": 0.0,
            "offpolicy/adv_mean": 0.0,
            "offpolicy/logprob_mean": 0.0,
        }

    log_prob_chunks = []
    position_chunks = []
    for step_idx_tensor in torch.unique(flat_step_indices[valid], sorted=True):
        step_idx = int(step_idx_tensor.item())
        step_mask = valid & (flat_step_indices == step_idx)
        step_positions = torch.where(step_mask)[0]
        neg_embeds = None
        if negative_text_embeddings is not None:
            neg_embeds = negative_text_embeddings.expand(step_positions.numel(), -1, -1)
        log_prob, _, _, _ = compute_transition_log_prob(
            model=model,
            x_t=flat_x_t[step_positions],
            x_t_minus_1=flat_x_t_minus_1[step_positions],
            step_idx=step_idx,
            text_embeddings=flat_text_embeddings[step_positions],
            weight_dtype=weight_dtype,
            sigmas_schedule=sigmas_schedule,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_text_embeddings=neg_embeds,
            start_frame_latents=None,
            sp_group=sp_group,
        )
        log_prob_chunks.append(log_prob.float())
        position_chunks.append(step_positions)

    selected_positions = torch.cat(position_chunks, dim=0)
    flat_log_probs = torch.cat(log_prob_chunks, dim=0)
    flat_step_indices = flat_step_indices[selected_positions]
    valid = torch.ones_like(flat_step_indices, dtype=torch.bool)

    flat_advantages = (
        trace_advantages[:, None]
        .expand(bsz, num_steps)
        .reshape(-1)
        .to(device=flat_log_probs.device, dtype=torch.float32)
    )[selected_positions]
    flat_advantages = torch.clamp(flat_advantages, -adv_clip_max, adv_clip_max)
    if positive_only:
        active = valid & (flat_advantages > 0)
        flat_advantages = flat_advantages.clamp_min(0.0)
    else:
        active = valid & torch.isfinite(flat_advantages) & (flat_advantages != 0)

    if not active.any():
        zero = flat_log_probs.sum() * 0.0
        return loss_weight * zero, {
            "offpolicy/loss_raw": 0.0,
            "offpolicy/loss_weighted": 0.0,
            "offpolicy/active_ratio": 0.0,
            "offpolicy/ratio_mean": 0.0,
            "offpolicy/adv_mean": float(flat_advantages.mean().detach().cpu()),
            "offpolicy/logprob_mean": float(flat_log_probs[valid].mean().detach().cpu()),
        }

    if use_teacher_logprob:
        flat_teacher_log_probs = trace_batch["trace_teacher_logprobs"].reshape(-1).to(
            device=flat_log_probs.device,
            dtype=torch.float32,
        )[selected_positions]
        finite_teacher = torch.isfinite(flat_teacher_log_probs)
        active = active & finite_teacher
        log_ratio = flat_log_probs - flat_teacher_log_probs
        ratio = torch.exp(torch.clamp(log_ratio, -ratio_clip, ratio_clip))
    else:
        flat_teacher_log_probs = torch.zeros_like(flat_log_probs)
        log_ratio = torch.zeros_like(flat_log_probs)
        ratio = torch.ones_like(flat_log_probs)

    if not active.any():
        zero = flat_log_probs.sum() * 0.0
        return loss_weight * zero, {
            "offpolicy/loss_raw": 0.0,
            "offpolicy/loss_weighted": 0.0,
            "offpolicy/active_ratio": 0.0,
            "offpolicy/ratio_mean": 0.0,
            "offpolicy/adv_mean": float(flat_advantages.mean().detach().cpu()),
            "offpolicy/logprob_mean": float(flat_log_probs[valid].mean().detach().cpu()),
        }

    step_weights = _step_weights_from_indices(
        flat_step_indices,
        sigmas_schedule=sigmas_schedule,
        num_inference_steps=num_inference_steps,
        mode=step_weighting,
        gamma=step_weight_gamma,
    ).to(flat_log_probs.device)

    surrogate = (surrogate or "detached_logprob").lower()
    if surrogate == "ratio":
        per_transition_loss = -flat_advantages * step_weights * ratio
    elif surrogate == "detached_logprob":
        per_transition_loss = -flat_advantages * step_weights * ratio.detach() * flat_log_probs
    else:
        raise ValueError(
            "rl_config.offpolicy_guidance.surrogate must be 'detached_logprob' or "
            f"'ratio', got {surrogate!r}"
        )

    raw_loss = per_transition_loss[active].mean()
    loss = loss_weight * raw_loss
    metrics = {
        "offpolicy/loss_raw": float(raw_loss.detach().cpu()),
        "offpolicy/loss_weighted": float(loss.detach().cpu()),
        "offpolicy/active_ratio": float(active.float().mean().detach().cpu()),
        "offpolicy/ratio_mean": float(ratio[active].mean().detach().cpu()),
        "offpolicy/ratio_max": float(ratio[active].max().detach().cpu()),
        "offpolicy/logratio_mean": float(log_ratio[active].mean().detach().cpu()),
        "offpolicy/adv_mean": float(flat_advantages[active].mean().detach().cpu()),
        "offpolicy/logprob_mean": float(flat_log_probs[active].mean().detach().cpu()),
        "offpolicy/teacher_logprob_mean": float(flat_teacher_log_probs[active].mean().detach().cpu()),
        "offpolicy/step_weight_mean": float(step_weights[active].mean().detach().cpu()),
    }
    return loss, metrics


def compute_gt_teacher_forcing_loss(
    model,
    scheduler,
    gt_latents,
    text_embeddings,
    gt_advantages,
    weight_dtype,
    loss_weight=0.05,
    adv_clip_max=5.0,
    positive_only=True,
    sp_group=None,
):
    """Advantage-weighted diffusion teacher forcing on GT videos.

    This is intentionally a supervised flow-matching branch, not an off-policy
    trajectory policy-gradient branch. The GT clean latent is noised with the
    training scheduler, and the model learns the original flow target
    `prior_dist - gt_latents`, weighted by the GT group's advantage.
    """
    if gt_latents.numel() == 0:
        return gt_latents.new_zeros(()), {}

    q_sample_results = scheduler.q_sample(gt_latents.float())
    interpolated_latents = q_sample_results["x_t"]
    prior_dist = q_sample_results["prior_dist"]
    sigmas = q_sample_results["sigmas"]
    timesteps = q_sample_results["timesteps"]

    if sp_group is not None:
        torch.distributed.broadcast(interpolated_latents, src=dist.get_global_rank(sp_group, 0), group=sp_group)
        torch.distributed.broadcast(prior_dist, src=dist.get_global_rank(sp_group, 0), group=sp_group)
        torch.distributed.broadcast(sigmas, src=dist.get_global_rank(sp_group, 0), group=sp_group)
        torch.distributed.broadcast(timesteps, src=dist.get_global_rank(sp_group, 0), group=sp_group)

    model_output = model(
        interpolated_latents.to(weight_dtype),
        timesteps.to(gt_latents.device),
        text_embeddings,
        start_frame_latents=None,
    )
    if isinstance(model_output, DTensor):
        model_output = model_output.full_tensor()

    target = prior_dist.float() - gt_latents.float()
    per_sample_loss = (model_output.float() - target).pow(2).mean(dim=tuple(range(1, target.ndim)))

    adv = gt_advantages.float()
    if adv.dim() > 1:
        adv = adv.mean(dim=tuple(range(1, adv.ndim)))
    adv = torch.clamp(adv, -adv_clip_max, adv_clip_max)
    if positive_only:
        active = adv > 0
        if not active.any():
            loss = loss_weight * per_sample_loss.sum() * 0.0
            metrics = {
                "gt_teacher/loss_raw": 0.0,
                "gt_teacher/loss_weighted": 0.0,
                "gt_teacher/adv_mean": float(adv.mean().detach().cpu()),
                "gt_teacher/active_ratio": 0.0,
            }
            return loss, metrics
        weighted_loss = (per_sample_loss[active] * adv[active].detach()).mean()
        active_ratio = active.float().mean()
    else:
        weighted_loss = (per_sample_loss * adv.detach()).mean()
        active_ratio = torch.ones_like(adv).mean()

    loss = loss_weight * weighted_loss
    metrics = {
        "gt_teacher/loss_raw": float(weighted_loss.detach().cpu()),
        "gt_teacher/loss_weighted": float(loss.detach().cpu()),
        "gt_teacher/adv_mean": float(adv.mean().detach().cpu()),
        "gt_teacher/active_ratio": float(active_ratio.detach().cpu()),
    }
    return loss, metrics


class TextPromptDataset(Dataset):
    def __init__(self, file_path, text_tokenizer_path, text_max_length=512, return_prompt_mask=True):
        with open(file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]
        self.text_processor = WanTextProcessor(
            tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
            model_max_length=text_max_length,
            return_prompt_mask=return_prompt_mask,
        )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        prompt_ids, prompt_mask = self.text_processor(prompt)
        return {
            PROMPT: prompt,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
            "metadata": {},
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [example[PROMPT] for example in examples]
        prompt_ids = torch.cat([example[PROMPT_IDS] for example in examples], dim=0)
        prompt_mask = torch.cat([example[PROMPT_MASK] for example in examples], dim=0)
        metadatas = [example["metadata"] for example in examples]
        return {
            PROMPT: prompts,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
            "metadata": metadatas,
        }


class TextPromptWithGTVideoDataset(Dataset):
    """Prompt dataset that additionally provides a ground-truth (GT) video per prompt.

    The preferred input format is:
        - ``prompt_file``: a txt file with one prompt per line
        - ``gt_video_dir``: a directory containing ``video_{line_idx}.mp4``

    Blank prompt lines are skipped, but the original physical line index is
    still used to resolve the GT video filename.

    For backward compatibility, ``metafile`` may also be provided. Each entry is
    a JSON object with at least:
        - "prompt" / "cap": the text prompt
        - "video_path" / "path": path to the GT video file
    Any extra keys (e.g. "fps", "num_frames", "start_frame_idx", "crop") are
    forwarded to ``WanVideoProcessor`` as ``meta_info``.

    The GT video is decoded into a ``(C, T, H, W)`` tensor normalized to ``[-1, 1]``
    (the same convention the VAE expects). It is used during RL rollout to inject
    an off-policy guidance trajectory whose reward joins the GRPO group, but which
    never participates in gradient / parameter updates.
    """

    def __init__(
        self,
        text_tokenizer_path,
        prompt_file=None,
        gt_video_dir=None,
        metafile=None,
        gt_filename_template="video_{}.mp4",
        text_max_length=512,
        return_prompt_mask=True,
        sample_height=480,
        sample_width=832,
        sample_num_frames=49,
        train_fps=16,
        sample_stride=None,
        force_cut_video_from_start=True,
    ):
        self.records = self._load_records(
            prompt_file=prompt_file,
            gt_video_dir=gt_video_dir,
            metafile=metafile,
            gt_filename_template=gt_filename_template,
        )
        self.text_processor = WanTextProcessor(
            tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
            model_max_length=text_max_length,
            return_prompt_mask=return_prompt_mask,
        )
        self.video_processor = WanVideoProcessor(
            sample_height=sample_height,
            sample_width=sample_width,
            sample_num_frames=sample_num_frames,
            train_fps=train_fps,
            sample_stride=sample_stride,
            force_cut_video_from_start=force_cut_video_from_start,
        )

    @staticmethod
    def _load_records(prompt_file=None, gt_video_dir=None, metafile=None, gt_filename_template="video_{}.mp4"):
        if metafile is not None:
            return TextPromptWithGTVideoDataset._load_metafile(metafile)

        if prompt_file is None or gt_video_dir is None:
            raise ValueError(
                "TextPromptWithGTVideoDataset requires either `metafile`, or both "
                "`prompt_file` and `gt_video_dir`."
            )

        records = []
        with open(prompt_file, "r") as f:
            for line_idx, line in enumerate(f):
                prompt = line.strip()
                if not prompt:
                    continue
                video_path = os.path.join(gt_video_dir, gt_filename_template.format(line_idx))
                if not os.path.exists(video_path):
                    raise FileNotFoundError(
                        f"GT video for prompt line {line_idx} not found: {video_path}"
                    )
                records.append({
                    "prompt": prompt,
                    "video_path": video_path,
                    "line_idx": line_idx,
                    "group_key": f"line:{line_idx}",
                })
        if not records:
            raise ValueError(f"No valid prompts found in prompt_file {prompt_file}")
        return records

    @staticmethod
    def _load_metafile(metafile):
        records = []
        with open(metafile, "r") as f:
            for row_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", obj.get("cap", None))
                video_path = obj.get("video_path", obj.get("path", None))
                if prompt is None or video_path is None:
                    raise ValueError(
                        f"Each metafile entry must contain a prompt and a video path, got: {obj}"
                    )
                obj.setdefault("line_idx", row_idx)
                obj.setdefault("group_key", f"line:{obj['line_idx']}")
                records.append(obj)
        if not records:
            raise ValueError(f"No valid records found in metafile {metafile}")
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        prompt = record.get("prompt", record.get("cap"))
        video_path = record.get("video_path", record.get("path"))
        prompt_ids, prompt_mask = self.text_processor(prompt)
        # WanVideoProcessor returns (C, T, H, W) normalized to [-1, 1].
        gt_video = self.video_processor(video_path, record)
        return {
            PROMPT: prompt,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
            VIDEO: gt_video,
            "metadata": {"line_idx": record.get("line_idx"), "video_path": video_path},
            "group_key": record.get("group_key", prompt),
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [example[PROMPT] for example in examples]
        prompt_ids = torch.cat([example[PROMPT_IDS] for example in examples], dim=0)
        prompt_mask = torch.cat([example[PROMPT_MASK] for example in examples], dim=0)
        gt_videos = torch.stack([example[VIDEO] for example in examples], dim=0)
        metadatas = [example["metadata"] for example in examples]
        group_keys = [example["group_key"] for example in examples]
        return {
            PROMPT: prompts,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
            VIDEO: gt_videos,
            "metadata": metadatas,
            "group_key": group_keys,
        }


def _trace_reward_to_scalar(reward):
    if reward is None:
        raise ValueError("Off-policy trace does not contain a reward.")
    if isinstance(reward, dict):
        for key in ("avg", "mean", "score", "reward", "total"):
            if key in reward:
                return _trace_reward_to_scalar(reward[key])
        values = []
        for value in reward.values():
            try:
                values.append(_trace_reward_to_scalar(value))
            except (TypeError, ValueError):
                continue
        if values:
            return float(np.mean(values))
        raise ValueError(f"Cannot extract a scalar reward from trace reward dict: {reward}")
    if isinstance(reward, torch.Tensor):
        reward = reward.detach().cpu().float()
        return float(reward.mean().item())
    if isinstance(reward, np.ndarray):
        return float(np.asarray(reward, dtype=np.float32).mean())
    if isinstance(reward, (list, tuple)):
        return float(np.asarray(reward, dtype=np.float32).mean())
    return float(reward)


def _discover_trace_paths(trace_dir=None, trace_files=None):
    paths = []
    if trace_files is not None:
        if isinstance(trace_files, str):
            trace_files = [trace_files]
        paths.extend(trace_files)
    if trace_dir is not None:
        for name in sorted(os.listdir(trace_dir)):
            if name.endswith((".pt", ".pth")):
                paths.append(os.path.join(trace_dir, name))
    paths = [os.path.abspath(path) for path in paths]
    if not paths:
        raise ValueError(
            "rl_config.offpolicy_guidance requires at least one saved trace. "
            "Set `trace_dir` or `trace_files`."
        )
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing off-policy trace files: {missing[:5]}")
    return paths


class DenoisingTraceDataset(Dataset):
    """Dataset for traces saved by infer/infer_osp.py.

    Each item contains a full teacher rollout trajectory. The final reward is
    normalized into a trajectory-level advantage and broadcast to every saved
    denoising transition during loss computation.
    """

    def __init__(
        self,
        text_tokenizer_path,
        trace_dir=None,
        trace_files=None,
        text_max_length=512,
        return_prompt_mask=True,
        advantage_mode="global",
        max_steps=None,
        step_stride=1,
        require_teacher_logprob=True,
    ):
        self.trace_paths = _discover_trace_paths(trace_dir=trace_dir, trace_files=trace_files)
        self.text_processor = WanTextProcessor(
            tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
            model_max_length=text_max_length,
            return_prompt_mask=return_prompt_mask,
        )
        self.max_steps = None if max_steps is None else int(max_steps)
        self.step_stride = max(1, int(step_stride))
        self.require_teacher_logprob = require_teacher_logprob
        self.records = self._load_records()
        self._attach_advantages(advantage_mode)

    def _load_records(self):
        records = []
        for trace_path in self.trace_paths:
            trace = torch.load(trace_path, map_location="cpu")
            prompt = trace.get("prompt", "")
            if not prompt:
                raise ValueError(f"Trace {trace_path} does not contain a non-empty prompt.")
            reward = _trace_reward_to_scalar(trace.get("reward", None))
            records.append({
                "path": trace_path,
                "prompt": prompt,
                "reward": reward,
                "sample_index": trace.get("sample_index", len(records)),
            })
            del trace
        return records

    def _attach_advantages(self, advantage_mode):
        advantage_mode = (advantage_mode or "global").lower()
        rewards = np.asarray([record["reward"] for record in self.records], dtype=np.float32)
        global_mean = rewards.mean()
        global_std = rewards.std()
        if advantage_mode == "raw":
            advantages = rewards
        elif advantage_mode == "centered":
            advantages = rewards - global_mean
        elif advantage_mode == "global":
            advantages = (rewards - global_mean) / (global_std + 1e-4)
        elif advantage_mode == "per_prompt":
            advantages = np.zeros_like(rewards)
            prompt_to_indices = defaultdict(list)
            for idx, record in enumerate(self.records):
                prompt_to_indices[record["prompt"]].append(idx)
            for indices in prompt_to_indices.values():
                prompt_rewards = rewards[indices]
                if len(indices) > 1:
                    advantages[indices] = (
                        prompt_rewards - prompt_rewards.mean()
                    ) / (prompt_rewards.std() + 1e-4)
                else:
                    advantages[indices] = (prompt_rewards - global_mean) / (global_std + 1e-4)
        else:
            raise ValueError(
                "rl_config.offpolicy_guidance.advantage_mode must be one of "
                f"'global', 'per_prompt', 'centered', or 'raw', got {advantage_mode!r}"
            )
        for record, advantage in zip(self.records, advantages):
            record["advantage"] = float(advantage)

    def __len__(self):
        return len(self.records)

    def _select_steps(self, steps):
        selected_steps = steps[::self.step_stride]
        if self.max_steps is not None:
            selected_steps = selected_steps[:self.max_steps]
        if not selected_steps:
            raise ValueError("No denoising steps selected from off-policy trace.")
        return selected_steps

    def __getitem__(self, idx):
        record = self.records[idx]
        trace = torch.load(record["path"], map_location="cpu")
        prompt = record["prompt"]
        prompt_ids, prompt_mask = self.text_processor(prompt)
        steps = self._select_steps(trace["steps"])

        latents = []
        next_latents = []
        teacher_logprobs = []
        step_indices = []
        for step in steps:
            latents.append(step["x_t"].float())
            next_latents.append(step["x_t_minus_1"].float())
            step_indices.append(int(step["step_index"]))
            teacher_logprob = step.get("teacher_logprob", None)
            if teacher_logprob is None:
                if self.require_teacher_logprob:
                    raise ValueError(
                        f"Trace {record['path']} is missing teacher_logprob at step {step['step_index']}. "
                        "Regenerate traces with denoising_trace_config.save_teacher_logprob=True, "
                        "or set rl_config.offpolicy_guidance.use_teacher_logprob=False."
                    )
                teacher_logprobs.append(torch.tensor(float("nan"), dtype=torch.float32))
            else:
                teacher_logprobs.append(torch.as_tensor(teacher_logprob, dtype=torch.float32).mean())

        return {
            PROMPT: prompt,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
            "metadata": {
                "trace_path": record["path"],
                "sample_index": record["sample_index"],
                "reward": record["reward"],
            },
            "trace_latents": torch.stack(latents, dim=0),
            "trace_next_latents": torch.stack(next_latents, dim=0),
            "trace_step_indices": torch.as_tensor(step_indices, dtype=torch.long),
            "trace_teacher_logprobs": torch.stack(teacher_logprobs, dim=0),
            "trace_advantage": torch.tensor(record["advantage"], dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [example[PROMPT] for example in examples]
        prompt_ids = torch.cat([example[PROMPT_IDS] for example in examples], dim=0)
        prompt_mask = torch.cat([example[PROMPT_MASK] for example in examples], dim=0)
        metadatas = [example["metadata"] for example in examples]
        return {
            PROMPT: prompts,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
            "metadata": metadatas,
            "trace_latents": torch.stack([example["trace_latents"] for example in examples], dim=0),
            "trace_next_latents": torch.stack([example["trace_next_latents"] for example in examples], dim=0),
            "trace_step_indices": torch.stack([example["trace_step_indices"] for example in examples], dim=0),
            "trace_teacher_logprobs": torch.stack(
                [example["trace_teacher_logprobs"] for example in examples], dim=0
            ),
            "trace_advantages": torch.stack([example["trace_advantage"] for example in examples], dim=0),
        }


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, \
            f"k cannot divide n*b, k={k}, num_replicas={num_replicas}, batch_size={batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, ref_prompts=None, ref_rewards=None, return_ref_advantages=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))
        # Inject off-policy GT (ground-truth) rewards into the per-prompt group
        # statistics. They shift the GRPO baseline (mean/std) but are NOT returned
        # as advantages, so they never drive gradient / parameter updates.
        global_ref_rewards = None
        ref_advantages = None
        if ref_prompts is not None and ref_rewards is not None:
            ref_prompts = np.array(ref_prompts)
            global_ref_rewards = np.array(ref_rewards, dtype=np.float64)
            ref_advantages = np.zeros_like(global_ref_rewards)
            for prompt in np.unique(ref_prompts):
                if prompt not in self.stats:
                    self.stats[prompt] = []
                self.stats[prompt].extend(global_ref_rewards[ref_prompts == prompt])
                self.history_prompts.add(hash(prompt))
        prompts_to_finalize = unique
        if ref_prompts is not None and ref_rewards is not None:
            prompts_to_finalize = np.unique(np.concatenate([unique, np.unique(ref_prompts)]))
        for prompt in prompts_to_finalize:
            self.stats[prompt] = np.stack(self.stats[prompt])
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std_source = rewards
                if global_ref_rewards is not None:
                    std_source = np.concatenate([rewards, global_ref_rewards], axis=0)
                std = np.std(std_source, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        if return_ref_advantages and ref_prompts is not None and global_ref_rewards is not None:
            for prompt in np.unique(ref_prompts):
                prompt_ref_rewards = global_ref_rewards[ref_prompts == prompt]
                mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
                if self.global_std:
                    std_source = np.concatenate([rewards, global_ref_rewards], axis=0)
                    std = np.std(std_source, axis=0, keepdims=True) + 1e-4
                else:
                    std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
                ref_advantages[ref_prompts == prompt] = (prompt_ref_rewards - mean) / std
            return advantages, ref_advantages
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        return avg_group_size, len(self.history_prompts)

    def clear(self):
        self.stats = {}


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    return zero_std_count / len(prompt_std_devs)


LORA_CHECKPOINT_PREFIX = "lora-checkpoint-"


def _lora_checkpoint_dir(save_dir, global_step, suffix: str = ""):
    return os.path.join(save_dir, f"{LORA_CHECKPOINT_PREFIX}{global_step}{suffix}")


def save_lora_checkpoint(model, save_dir, global_step, suffix: str = ""):
    """Save LoRA-only adapter weights.

    Base model parameters are frozen during RL training, so we deliberately
    skip persisting them — only the LoRA matrices (and adapter config) are
    written out. Use ``suffix='-ema'`` to save the EMA-shadowed LoRA.
    """
    save_root = _lora_checkpoint_dir(save_dir, global_step, suffix)
    if dist.get_rank() == 0:
        os.makedirs(save_root, exist_ok=True)
    dist.barrier()
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            if isinstance(param, DTensor):
                full_param = param.full_tensor()
            else:
                full_param = param

            if dist.get_rank() == 0:
                lora_state_dict[name] = full_param.detach().clone().cpu()

    if dist.get_rank() == 0 and lora_state_dict:
        torch.save(lora_state_dict, os.path.join(save_root, "adapter_model.bin"))
        if hasattr(model, 'peft_config'):
            for adapter_name, peft_cfg in model.peft_config.items():
                config_dict = peft_cfg.to_dict() if hasattr(peft_cfg, 'to_dict') else vars(peft_cfg)
                with open(os.path.join(save_root, "adapter_config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2, default=str)
                break
        
        print(f"[Rank 0] LoRA checkpoint saved to {save_root} ({len(lora_state_dict)} parameters)")


RL_TRAINING_STATE_FILE = "rl_training_state.json"


def save_rl_training_state(save_dir, global_step, next_epoch):
    """Persist RL resume metadata next to the LoRA checkpoint."""
    save_root = _lora_checkpoint_dir(save_dir, global_step)
    if dist.get_rank() == 0:
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, RL_TRAINING_STATE_FILE), "w", encoding="ascii") as f:
            json.dump(
                {
                    "global_step": int(global_step),
                    "next_epoch": int(next_epoch),
                },
                f,
                indent=2,
            )
    dist.barrier()


def load_rl_training_state(lora_dir):
    """Load RL resume metadata for epoch/global_step bookkeeping."""
    state_path = os.path.join(lora_dir, RL_TRAINING_STATE_FILE)
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="ascii") as f:
        return json.load(f)



# ==================== Main Training ====================

def main(config):
    logger = get_logger()

    # ========== Config ==========
    seed = config.get("seed", 42)

    # model config
    model_name = config.get("model_name", "osp_next")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})
    # skiparse 相关
    sparse_ratio = model_config.get("sparse_ratio", 1)
    skiparse_model_type = model_config.get("skiparse_model_type", "full")
    is_skiparse_model = skiparse_model_type != "full"
    num_full_blocks = model_config.get("num_full_blocks", 0)

    # LoRA config
    lora_config = config.get("lora_config", {})
    lora_rank = lora_config.get("rank", 32)
    lora_alpha = lora_config.get("alpha", 64)
    lora_target_modules = lora_config.get("target_modules", [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    ])
    lora_path = lora_config.get("lora_path", None)

    # RL config
    rl_config = config.get("rl_config", {})
    num_inference_steps = rl_config.get("num_inference_steps", 20)
    guidance_scale = _as_float(rl_config.get("guidance_scale", 5.0), "rl_config.guidance_scale")
    sample_batch_size = rl_config.get("sample_batch_size", 4)
    train_batch_size = rl_config.get("train_batch_size", 4)
    num_batches_per_epoch = rl_config.get("num_batches_per_epoch", 4)
    num_inner_epochs = rl_config.get("num_inner_epochs", 1)
    num_image_per_prompt = rl_config.get("num_image_per_prompt", 4)
    sample_time_per_prompt = rl_config.get("sample_time_per_prompt", 1)
    timestep_fraction = _as_float(rl_config.get("timestep_fraction", 1.0), "rl_config.timestep_fraction")
    clip_range = _as_float(rl_config.get("clip_range", 5e-3), "rl_config.clip_range")
    adv_clip_max = _as_float(rl_config.get("adv_clip_max", 5.0), "rl_config.adv_clip_max")
    kl_reward = _as_float(rl_config.get("kl_reward", 0.0), "rl_config.kl_reward")
    kl_beta = _as_float(rl_config.get("kl_beta", 0.0), "rl_config.kl_beta")
    use_cfg_in_train = rl_config.get("use_cfg_in_train", True)
    per_prompt_stat_tracking = rl_config.get("per_prompt_stat_tracking", True)
    global_std = rl_config.get("global_std", False)
    reward_fn_config = rl_config.get("reward_fn", {})
    prompt_file = rl_config.get("prompt_file", None)
    eval_prompt_file = rl_config.get("eval_prompt_file", None)
    video_height = rl_config.get("height", 720)
    video_width = rl_config.get("width", 1280)
    video_num_frames = rl_config.get("num_frames", 81)
    eval_freq = rl_config.get("eval_freq", 10000)
    eval_num_steps = rl_config.get("eval_num_steps", 50)
    # SDE/ODE hybrid: 前 sde_steps 步使用 SDE（有噪声），剩余步使用 ODE（确定性）
    sde_steps = rl_config.get("sde_steps", num_inference_steps)  # 默认全 SDE

    # GT (off-policy) guidance config: by default, `prompt_file` line N is paired
    # with `gt_video_dir/video_N.mp4`. During rollout the GT video reward joins
    # the GRPO group to guide the advantage baseline, but it never participates
    # in the gradient / parameter update.
    gt_guidance_config = rl_config.get("gt_guidance", {})
    use_gt_guidance = gt_guidance_config.get("enable", False)
    gt_prompt_file = gt_guidance_config.get("prompt_file", prompt_file)
    gt_video_dir = gt_guidance_config.get("gt_video_dir", gt_guidance_config.get("video_dir", None))
    gt_metafile = gt_guidance_config.get("metafile", None)
    gt_filename_template = gt_guidance_config.get("filename_template", "video_{}.mp4")
    gt_train_fps = gt_guidance_config.get("train_fps", 16)
    gt_sample_stride = gt_guidance_config.get("sample_stride", None)
    gt_force_cut_video_from_start = gt_guidance_config.get("force_cut_video_from_start", True)
    if use_gt_guidance and gt_metafile is None and (gt_prompt_file is None or gt_video_dir is None):
        raise ValueError(
            "When rl_config.gt_guidance.enable is True, set either "
            "rl_config.gt_guidance.metafile, or set rl_config.prompt_file plus "
            "rl_config.gt_guidance.gt_video_dir."
        )

    # Stage-2 LUFFY-style improvement: use the GT sample's group advantage to
    # weight the original diffusion/flow-matching teacher-forcing loss.
    gt_teacher_config = rl_config.get("gt_teacher_forcing", {})
    use_gt_teacher_forcing = use_gt_guidance and gt_teacher_config.get("enable", False)
    gt_teacher_loss_weight = _as_float(
        gt_teacher_config.get("loss_weight", gt_teacher_config.get("lambda", 0.05)),
        "rl_config.gt_teacher_forcing.loss_weight",
    )
    gt_teacher_adv_clip_max = _as_float(
        gt_teacher_config.get("adv_clip_max", adv_clip_max),
        "rl_config.gt_teacher_forcing.adv_clip_max",
    )
    gt_teacher_positive_only = gt_teacher_config.get("positive_only", True)
    gt_teacher_train_batch_size = max(1, int(gt_teacher_config.get("train_batch_size", train_batch_size)))
    gt_teacher_every_n_steps = max(1, int(gt_teacher_config.get("every_n_steps", 1)))

    # Stage-3 LUFFY-style off-policy guidance: reuse teacher rollout
    # transitions saved by infer/infer_osp.py. The current policy is trained to
    # increase the likelihood of high-advantage teacher transitions
    # p_theta(x_{t-1} | x_t, prompt), with optional importance correction against
    # the saved teacher transition logprob.
    offpolicy_config = rl_config.get("offpolicy_guidance", {})
    use_offpolicy_guidance = bool(offpolicy_config.get("enable", False))
    offpolicy_trace_dir = offpolicy_config.get("trace_dir", offpolicy_config.get("denoising_trace_dir", None))
    offpolicy_trace_files = offpolicy_config.get("trace_files", None)
    offpolicy_loss_weight = _as_float(
        offpolicy_config.get("loss_weight", offpolicy_config.get("lambda", 0.01)),
        "rl_config.offpolicy_guidance.loss_weight",
    )
    offpolicy_adv_clip_max = _as_float(
        offpolicy_config.get("adv_clip_max", adv_clip_max),
        "rl_config.offpolicy_guidance.adv_clip_max",
    )
    offpolicy_ratio_clip = _as_float(
        offpolicy_config.get("ratio_clip", 5.0),
        "rl_config.offpolicy_guidance.ratio_clip",
    )
    offpolicy_positive_only = bool(offpolicy_config.get("positive_only", True))
    offpolicy_use_teacher_logprob = bool(offpolicy_config.get("use_teacher_logprob", True))
    offpolicy_surrogate = offpolicy_config.get("surrogate", "detached_logprob")
    offpolicy_advantage_mode = offpolicy_config.get("advantage_mode", "global")
    offpolicy_train_batch_size = max(1, int(offpolicy_config.get("train_batch_size", offpolicy_config.get("batch_size", train_batch_size))))
    offpolicy_every_n_steps = max(1, int(offpolicy_config.get("every_n_steps", 1)))
    offpolicy_num_workers = max(0, int(offpolicy_config.get("num_workers", 0)))
    offpolicy_max_steps = offpolicy_config.get("max_steps", None)
    offpolicy_step_stride = max(1, int(offpolicy_config.get("step_stride", 1)))
    offpolicy_step_weighting = offpolicy_config.get("step_weighting", "uniform")
    offpolicy_step_weight_gamma = _as_float(
        offpolicy_config.get("step_weight_gamma", 1.0),
        "rl_config.offpolicy_guidance.step_weight_gamma",
    )
    num_train_timesteps = int(sde_steps * timestep_fraction)
    train_timestep_indices = list(range(num_train_timesteps))
    if offpolicy_max_steps is None:
        offpolicy_max_steps = num_train_timesteps

    debug_training_config = rl_config.get("debug_training", {})
    if isinstance(debug_training_config, bool):
        debug_training_config = {"enable": debug_training_config}
    debug_training = bool(debug_training_config.get("enable", False))
    debug_training_epochs = max(1, int(debug_training_config.get("epochs", 1)))
    debug_training_inner_epochs = max(1, int(debug_training_config.get("inner_epochs", 1)))
    debug_training_micro_batches = max(1, int(debug_training_config.get("micro_batches", 1)))
    debug_training_timesteps = max(1, int(debug_training_config.get("timesteps", 1)))
    debug_training_sync = bool(debug_training_config.get("sync", True))
    debug_stack_timeout = int(debug_training_config.get("stack_timeout_seconds", 600))

    # EMA config
    ema_decay = config.get("ema_decay", 0.9999)
    ema_update_interval = config.get("ema_update_interval", 1)

    # data config (for prompt dataset)
    data_config = config.get("data_config", {})

    # optimizer config
    optimizer_config = config.get("optimizer_config", {})

    # training config
    num_epochs = config.get("num_epochs", 1000)
    gradient_checkpointing = config.get("gradient_checkpointing", False)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    init_max_grad_norm = config.get("init_max_grad_norm", 1.0)
    save_interval = config.get("save_interval", 100)
    weight_dtype = config.get("weight_dtype", "bfloat16")
    resume_epoch = config.get("resume_epoch", None)
    resume_override_lr = optimizer_config.get("resume_override_lr", False)
    resume_lr = optimizer_config.get("resume_lr", None)
    if resume_lr is not None:
        resume_lr = _as_float(resume_lr, "optimizer_config.resume_lr")
    reshard_after_forward = config.get("reshard_after_forward", None)
    model_cpu_offload = config.get("model_cpu_offload", False)
    encoder_cpu_offload = config.get("encoder_cpu_offload", False)
    use_sequence_parallel = config.get("use_sequence_parallel", False)
    use_skiparse_sequence_parallel = config.get("use_skiparse_sequence_parallel", False)
    deterministic_training = config.get("deterministic_training", False)

    # save config
    output_dir = config.get("output_dir", "./output_rl_lora")
    save_with_dcp_api = config.get("save_with_dcp_api", False)

    setup_distributed_env()
    verify_min_gpu_count()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = str_to_precision(weight_dtype)

    if debug_training:
        faulthandler.enable()

    def should_debug_train(epoch_idx=None, inner_idx=None, micro_idx=None, timestep_idx=None):
        if not debug_training:
            return False
        if epoch_idx is not None and epoch_idx >= start_epoch + debug_training_epochs:
            return False
        if inner_idx is not None and inner_idx >= debug_training_inner_epochs:
            return False
        if micro_idx is not None and micro_idx >= debug_training_micro_batches:
            return False
        if timestep_idx is not None and timestep_idx >= debug_training_timesteps:
            return False
        return True

    def debug_train_log(stage, epoch_idx=None, inner_idx=None, micro_idx=None, timestep_idx=None, extra="", sync=False):
        if not should_debug_train(epoch_idx, inner_idx, micro_idx, timestep_idx):
            return
        arm_stack_timer = debug_stack_timeout > 0 and (stage.endswith("_before") or sync)
        if arm_stack_timer:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(debug_stack_timeout, repeat=True)
        context = []
        if epoch_idx is not None:
            context.append(f"epoch={epoch_idx}")
        if inner_idx is not None:
            context.append(f"inner={inner_idx}")
        if micro_idx is not None:
            context.append(f"mb={micro_idx}")
        if timestep_idx is not None:
            context.append(f"t={timestep_idx}")
        context = " ".join(context)
        if extra:
            extra = f" | {extra}"
        print(
            f"[TRAIN-DBG][rank={rank}/{world_size}][local={local_rank}] "
            f"{stage} {context}{extra} | mem={get_memory_allocated()}GiB",
            flush=True,
        )
        if sync:
            torch.cuda.synchronize()
            print(
                f"[TRAIN-DBG][rank={rank}/{world_size}][local={local_rank}] "
                f"{stage}:sync_done {context} | mem={get_memory_allocated()}GiB",
                flush=True,
            )
        if debug_stack_timeout > 0 and not stage.endswith("_before"):
            faulthandler.cancel_dump_traceback_later()

    wandb_config = config.get("wandb_config", {})
    if wandb_config.get("project_name", None) is not None and rank == 0:
        project_name = wandb_config.get("project_name")
        wandb.init(
            project=project_name,
            name=wandb_config.get("exp_name", project_name),
            config=config,
            dir=output_dir,
        )

    fsdp_size = config.get("fsdp_size", 8)
    if fsdp_size > world_size:
        fsdp_size = world_size
        log_on_main_process(logger, f"Warning: GPU nums not enough! FSDP size reset to {fsdp_size}!")
    elif world_size % fsdp_size != 0:
        raise ValueError(f"world_size % fsdp_size != 0, fsdp error!")
    ddp_size = config.get("ddp_size", world_size // fsdp_size)
    ddp_fsdp_mesh = init_device_mesh("cuda", (ddp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp"))
    logger.info(f"rank {rank} use ddp mesh {ddp_fsdp_mesh['ddp']} and fsdp mesh {ddp_fsdp_mesh['fsdp']}")

    dp_group = dist.group.WORLD
    sp_size = 1
    use_sequence_parallel = use_sequence_parallel and config.get("sp_size", 1) > 1
    skiparse_sp_size = 1
    use_skiparse_sequence_parallel = use_skiparse_sequence_parallel and config.get("skiparse_sp_size", 1) > 1 and sparse_ratio > 1
    use_global_sequence_parallel = use_sequence_parallel or use_skiparse_sequence_parallel
    global_sp_size = 1
    full_sp_size = 1
    use_full_blocks_sequence_parallel = use_global_sequence_parallel and is_skiparse_model and num_full_blocks > 0
    global_sp_group = None

    if use_global_sequence_parallel:
        if use_sequence_parallel:
            sp_size = config.get("sp_size", 1)
        if use_skiparse_sequence_parallel:
            skiparse_sp_size = config.get("skiparse_sp_size", 1)
            if is_skiparse_model:
                # OSP-Next skiparse is 2D, so the per-rank shard must evenly divide sparse_ratio ** 2.
                assert skiparse_sp_size <= sparse_ratio ** 2 and (sparse_ratio ** 2) % skiparse_sp_size == 0
        global_sp_size = skiparse_sp_size * sp_size
        dp_global_sp_mesh = init_device_mesh("cuda", (world_size // global_sp_size, skiparse_sp_size, sp_size), mesh_dim_names=("dp", "skiparse_sp", "sp"))
        dp_group = dp_global_sp_mesh["dp"].get_group()
        global_sp_group = dp_global_sp_mesh["skiparse_sp", "sp"]._flatten().get_group()
        skiparse_sp_group = dp_global_sp_mesh["skiparse_sp"].get_group()
        full_sp_group = sp_group = dp_global_sp_mesh["sp"].get_group()
        log_on_main_process(logger, f"Using Sequence parallel: global_sp_size={global_sp_size}, sp_size={sp_size}, skiparse_sp_size={skiparse_sp_size}")
        sp_state.reset(global_sp_group=global_sp_group, sp_group=sp_group, skiparse_sp_group=skiparse_sp_group, full_sp_group=full_sp_group)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    set_seed(seed, device_specific=False)

    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device,
    )
    log_on_main_process(logger, f"VAE model initialized, memory: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing text encoder model...")
    text_encoder_device_mesh = None
    if text_encoder_config.get("use_fsdp", False):
        num_replicate = max(world_size // 8, 1)
        num_shard = world_size // num_replicate
        text_encoder_device_mesh = init_device_mesh("cuda", (num_replicate, num_shard), mesh_dim_names=("replicate", "shard"))
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device,
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        use_fsdp=text_encoder_config.get("use_fsdp", False),
        device_mesh=text_encoder_device_mesh,
    )
    log_on_main_process(logger, f"Text encoder initialized, memory: {get_memory_allocated()} GiB")

    text_encoder_use_fsdp = text_encoder_config.get("use_fsdp", False)
    if encoder_cpu_offload:
        log_on_main_process(logger, "Offloading VAE and text encoder to CPU to save GPU memory...")
        vae.model.to("cpu")
        if not text_encoder_use_fsdp:
            text_encoder.model.to("cpu")
        torch.cuda.empty_cache()
        log_on_main_process(logger, f"After encoder CPU offload, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing scheduler...")
    scheduler = schedulers[scheduler_config.get("scheduler_name", "flow_matching")](**scheduler_config)

    log_on_main_process(logger, "Initializing diffusion model...")
    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)
    has_loaded_pretrained_model = False

    if pretrained_model_dir_or_checkpoint is not None and os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_dir {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
        has_loaded_pretrained_model = True
    elif pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load base model from checkpoint file {pretrained_model_dir_or_checkpoint}")
        model = models[model_name](**model_config)
        if pretrained_model_dir_or_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load
            full_sd = safe_load(pretrained_model_dir_or_checkpoint, device="cpu")
        else:
            full_sd = torch.load(pretrained_model_dir_or_checkpoint, mmap=True, weights_only=True, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(full_sd, strict=False)
        if rank == 0:
            if missing_keys:
                print(f"[Base model checkpoint] missing_keys: {missing_keys}")
            if unexpected_keys:
                print(f"[Base model checkpoint] unexpected_keys: {unexpected_keys}")
        del full_sd
        has_loaded_pretrained_model = True
    else:
        log_on_main_process(logger, "Init model from scratch")
        with torch.device("meta"):
            model = models[model_name](**model_config)

    if use_sequence_parallel or use_full_blocks_sequence_parallel:
        if use_sequence_parallel and model.num_heads % sp_size != 0:
            raise ValueError(f"When using Sequence parallel, num_heads {model.num_heads} must be multiple of sp_size {sp_size}!")
        if use_full_blocks_sequence_parallel:
            if global_sp_size <= model.num_heads and model.num_heads % global_sp_size == 0:
                full_sp_size = global_sp_size
            else:
                gcd = math.gcd(model.num_heads, global_sp_size)
                full_sp_size = gcd
            dummy_mesh = init_device_mesh("cuda", (world_size // full_sp_size, full_sp_size), mesh_dim_names=("dummy", "full_sp"))
            full_sp_group = dummy_mesh["full_sp"].get_group()
            sp_state.reset(full_sp_group=full_sp_group)

    if lora_path:
        log_on_main_process(logger, f"Loading existing LoRA from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model.set_adapter("default")
    else:
        log_on_main_process(logger, f"Initializing new LoRA with rank={lora_rank}, alpha={lora_alpha}")
        log_on_main_process(logger, f"LoRA target modules: {lora_target_modules}")
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, peft_config)

    if rank == 0:
        model.print_trainable_parameters()

    for name, param in model.named_parameters():
        param.requires_grad = 'lora_' in name

    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    model.train()

    if model_cpu_offload:
        log_on_main_process(logger, "Moving model to CPU for FSDP CPU offloading to prevent NPU OOM...")
        model.to("cpu")
        torch.cuda.empty_cache()

    log_on_main_process(logger, "Starting FSDP2 wrapping...")
    FSDP2_mix_wrapper(
        model,
        dp_mesh=ddp_fsdp_mesh,
        weight_dtype=weight_dtype,
        main_block_to_half=models_main_block[model_name],
        blocks_to_float=models_blocks_to_float[model_name],
        blocks_to_output_float=models_blocks_to_output_float[model_name],
        reshard_after_forward=reshard_after_forward,
        cpu_offload=model_cpu_offload,
    )
    log_on_main_process(logger, "FSDP2 wrapping completed successfully.")

    if not has_loaded_pretrained_model:
        init_device = "cpu" if model_cpu_offload else device
        model.to_empty(device=init_device)
        set_seed(seed, device_specific=False)
        base_model.reset_parameters()

    log_on_main_process(logger, f"Diffusion model (LoRA + FSDP2) initialized, memory: {get_memory_allocated()} GiB")

    if gradient_checkpointing:
        log_on_main_process(logger, "Using gradient checkpointing")
        if hasattr(base_model, 'set_gradient_checkpointing'):
            base_model.set_gradient_checkpointing(True)
        elif hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

    if (kl_reward > 0 or kl_beta > 0) and hasattr(model, 'disable_adapter'):
        try:
            with model.disable_adapter():
                log_on_main_process(logger, "disable_adapter() Sequence manager works under FSDP2.")
        except Exception as e:
            log_on_main_process(logger, f"WARNING: disable_adapter() failed under FSDP2: {e}")
            log_on_main_process(logger, "KL computation may not work correctly. Consider using a separate ref_model.")

    log_on_main_process(logger, "Initializing EMA model...")
    ema_model = EMAModel(model, decay=ema_decay, update_interval=ema_update_interval)
    _lora_ema_keys = {n for n, _ in model.named_parameters() if 'lora_' in n}

    @torch.no_grad()
    def _lora_only_ema_update(model, step):
        if step % ema_model.update_interval != 0:
            return
        for name, param in model.named_parameters():
            shadow_param = ema_model.shadow_params[name]
            if name in _lora_ema_keys:
                shadow_param.data.sub_(
                    ema_model.one_minus_decay * (shadow_param.data.float() - param.data.float())
                )
            else:
                shadow_param.data.copy_(param.data)

    ema_model.update = _lora_only_ema_update
    log_on_main_process(logger, f"EMA model initialized (LoRA-only update, {len(_lora_ema_keys)} LoRA keys), memory: {get_memory_allocated()} GiB")

    # RL training only persists LoRA adapters (base model is frozen), so there is
    # no full FSDP checkpoint to resume from. To resume, set `lora_config.lora_path`
    # in the config to a previously saved `lora-checkpoint-{step}/` directory;
    # the LoRA weights are loaded above via `PeftModel.from_pretrained`, and any
    # sidecar metadata (epoch / grad clipper state) is restored below.
    if not has_loaded_pretrained_model:
        log_on_main_process(logger, f"Warning! Training from scratch, pretrained_model_dir_or_checkpoint={pretrained_model_dir_or_checkpoint}")

    log_on_main_process(logger, "Initializing optimizer...")
    learning_rate = _as_float(optimizer_config.get("lr", 5e-4), "optimizer_config.lr")
    weight_decay_val = _as_float(optimizer_config.get("weight_decay", 1e-2), "optimizer_config.weight_decay")
    lora_param_names = {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"}
    trainable_parameters = [
        p for n, p in model.named_parameters()
        if any(lora_key in n for lora_key in lora_param_names)
    ]
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        lora_params = sum(p.numel() for p in trainable_parameters)
        log_on_main_process(logger, f"Optimizer: {lora_params:,} LoRA params / {total_params:,} total params")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        betas=optimizer_config.get("betas", (0.9, 0.999)),
        weight_decay=weight_decay_val,
        eps=_as_float(optimizer_config.get("eps", 1e-15), "optimizer_config.eps"),
    )
    adaptive_grad_clipper = AdaptiveGradClipper(
        init_max_grad_norm=init_max_grad_norm,
        model_parallel_group=ddp_fsdp_mesh["fsdp"].get_group(),
    )

    # Restore adaptive grad clipper state and epoch/global_step bookkeeping from
    # the LoRA checkpoint directory that `lora_path` points to (if any). Note:
    # AdamW optimizer state is NOT persisted across runs anymore — Adam moments
    # are small for LoRA and re-warm quickly, and skipping them lets us avoid
    # touching the base model.
    resume_global_step = 0
    start_epoch = 0
    rl_training_state = None
    if lora_path is not None:
        lora_state_dir = lora_path if os.path.isdir(lora_path) else os.path.dirname(lora_path)
        adaptive_grad_clipper.load(output_dir=lora_state_dir)
        rl_training_state = load_rl_training_state(lora_state_dir)
        override_lr = None
        if resume_lr is not None:
            override_lr = resume_lr
        elif resume_override_lr:
            override_lr = learning_rate
        if override_lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = override_lr
            log_on_main_process(
                logger,
                f"Resume LR override enabled, force optimizer lr to {override_lr}",
            )

    if rl_training_state is not None:
        resume_global_step = int(rl_training_state.get("global_step", 0))
        if resume_epoch is not None:
            start_epoch = int(resume_epoch)
            log_on_main_process(
                logger,
                f"Resume epoch manually specified as {start_epoch}. "
                "Skipping auto epoch recovery from rl_training_state.json.",
            )
        else:
            start_epoch = int(rl_training_state.get("next_epoch", 0))
    elif resume_epoch is not None:
        start_epoch = int(resume_epoch)
        log_on_main_process(
            logger,
            f"resume_epoch={start_epoch} is set but no rl_training_state.json was found "
            f"alongside lora_path={lora_path}. "
            "Training will start from scratch with the specified epoch index.",
        )

    set_seed(seed, device_specific=True, process_group=dp_group, deterministic=deterministic_training)

    log_on_main_process(logger, f"Initializing reward functions with config: {reward_fn_config}")
    import ospnext.rewards.rewards
    reward_fn = getattr(ospnext.rewards.rewards, 'multi_score')(device, reward_fn_config)
    dist.barrier()
    log_on_main_process(logger, "All ranks passed reward initialization.")

    text_tokenizer_path = data_config.get("dataset_config", {}).get("text_tokenizer_path", None)
    text_max_length = data_config.get("dataset_config", {}).get("tokenizer_max_length", text_encoder_config.get("text_len", 512))
    if text_tokenizer_path is None:
        raise ValueError("data_config.dataset_config.text_tokenizer_path must be specified.")
    if not use_gt_guidance and prompt_file is None:
        raise ValueError("prompt_file must be specified for RL training when gt_guidance is disabled.")

    if use_gt_guidance:
        log_on_main_process(
            logger,
            "GT guidance enabled. Loading prompt+GT-video dataset from "
            f"{gt_metafile if gt_metafile is not None else gt_prompt_file} "
            f"with GT videos from {gt_video_dir}",
        )
        train_dataset = TextPromptWithGTVideoDataset(
            text_tokenizer_path=text_tokenizer_path,
            prompt_file=gt_prompt_file,
            gt_video_dir=gt_video_dir,
            metafile=gt_metafile,
            gt_filename_template=gt_filename_template,
            text_max_length=text_max_length,
            sample_height=video_height,
            sample_width=video_width,
            sample_num_frames=video_num_frames,
            train_fps=gt_train_fps,
            sample_stride=gt_sample_stride,
            force_cut_video_from_start=gt_force_cut_video_from_start,
        )
        train_collate_fn = TextPromptWithGTVideoDataset.collate_fn
    else:
        train_dataset = TextPromptDataset(
            file_path=prompt_file,
            text_tokenizer_path=text_tokenizer_path,
            text_max_length=text_max_length,
        )
        train_collate_fn = TextPromptDataset.collate_fn

    dp_size = dp_group.size() if use_global_sequence_parallel else world_size
    dp_rank = dist.get_rank(dp_group) if use_global_sequence_parallel else rank

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=sample_batch_size,
        k=num_image_per_prompt,
        num_replicas=dp_size,
        rank=dp_rank,
        seed=seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=train_collate_fn,
    )

    offpolicy_dataloader = None
    offpolicy_sampler = None
    if use_offpolicy_guidance:
        offpolicy_dataset = DenoisingTraceDataset(
            text_tokenizer_path=text_tokenizer_path,
            trace_dir=offpolicy_trace_dir,
            trace_files=offpolicy_trace_files,
            text_max_length=text_max_length,
            advantage_mode=offpolicy_advantage_mode,
            max_steps=offpolicy_max_steps,
            step_stride=offpolicy_step_stride,
            require_teacher_logprob=offpolicy_use_teacher_logprob,
        )
        offpolicy_sampler = DistributedSampler(
            offpolicy_dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            drop_last=False,
        )
        offpolicy_dataloader = DataLoader(
            offpolicy_dataset,
            batch_size=offpolicy_train_batch_size,
            sampler=offpolicy_sampler,
            num_workers=offpolicy_num_workers,
            collate_fn=DenoisingTraceDataset.collate_fn,
            pin_memory=True,
        )
        offpolicy_rewards = np.asarray(
            [record["reward"] for record in offpolicy_dataset.records],
            dtype=np.float32,
        )
        offpolicy_advantages = np.asarray(
            [record["advantage"] for record in offpolicy_dataset.records],
            dtype=np.float32,
        )
        log_on_main_process(
            logger,
            f"Off-policy guidance loaded {len(offpolicy_dataset)} traces | "
            f"reward mean={offpolicy_rewards.mean():.4f}, std={offpolicy_rewards.std():.4f} | "
            f"adv mean={offpolicy_advantages.mean():.4f}, std={offpolicy_advantages.std():.4f}",
        )

    test_dataloader = None
    eval_sampler = None
    ddp_rank_for_eval, _, ddp_size_for_eval = get_ddp_rank_and_fsdp_local_rank(
        rank=rank,
        fsdp_size=fsdp_size,
        world_size=world_size,
    )

    if eval_prompt_file is not None:
        eval_dataset = TextPromptDataset(
            file_path=eval_prompt_file,
            text_tokenizer_path=text_tokenizer_path,
            text_max_length=text_max_length,
        )

        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=ddp_size_for_eval,
            rank=ddp_rank_for_eval,
            shuffle=False,
            drop_last=False,
        )

        test_dataloader = DataLoader(
            eval_dataset,
            batch_size=sample_batch_size,
            sampler=eval_sampler,
            collate_fn=TextPromptDataset.collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    if num_image_per_prompt * sample_time_per_prompt <= 1:
        per_prompt_stat_tracking = False
    stat_tracker = PerPromptStatTracker(global_std=global_std) if per_prompt_stat_tracking else None

    log_on_main_process(logger, "Computing negative text embedding...")
    from ospnext.utils.constant import NEGATIVE_PROMPT
    neg_text_processor = WanTextProcessor(
        tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
        model_max_length=text_max_length,
        return_prompt_mask=True,
    )
    neg_prompt_ids, neg_prompt_mask = neg_text_processor(NEGATIVE_PROMPT)
    neg_prompt_ids = neg_prompt_ids.to(device)
    neg_prompt_mask = neg_prompt_mask.to(device)
    if encoder_cpu_offload:
        if not text_encoder_use_fsdp:
            text_encoder.model.to(device)
    with torch.no_grad():
        neg_text_embeddings = text_encoder(neg_prompt_ids, neg_prompt_mask)
    if encoder_cpu_offload:
        if not text_encoder_use_fsdp:
            text_encoder.model.to("cpu")
        torch.cuda.empty_cache()

    sigmas_schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas_schedule = shift * sigmas_schedule / (1 + (shift - 1) * sigmas_schedule)

    def encode_text_batch_for_loss(prompt_ids, prompt_mask):
        if encoder_cpu_offload:
            if not text_encoder_use_fsdp:
                text_encoder.model.to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(prompt_ids.to(device), prompt_mask.to(device))
        if encoder_cpu_offload:
            if not text_encoder_use_fsdp:
                text_encoder.model.to("cpu")
            torch.cuda.empty_cache()
        return text_embeddings

    offpolicy_iterator = None

    def next_offpolicy_batch():
        nonlocal offpolicy_iterator
        if offpolicy_dataloader is None:
            return None
        if offpolicy_iterator is None:
            offpolicy_iterator = iter(offpolicy_dataloader)
        try:
            return next(offpolicy_iterator)
        except StopIteration:
            offpolicy_iterator = iter(offpolicy_dataloader)
            return next(offpolicy_iterator)

    executor = futures.ThreadPoolExecutor(max_workers=8)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log_on_main_process(logger, f"""
    {'=' * 20}Start RL Training (GRPO + LoRA + FSDP2){'=' * 20}
    Model: {model_name}
    LoRA rank: {lora_rank}, alpha: {lora_alpha}
    LoRA target modules: {lora_target_modules}
    Trainable parameters: {params_nums_to_str(trainable_params)} / {params_nums_to_str(total_params)}
    Scheduler: {scheduler_config.get("scheduler_name", "flow_matching")}
    Num epochs: {num_epochs}
    Num inference steps: {num_inference_steps}
    SDE steps: {sde_steps} (step 0~{sde_steps-1}: SDE, step {sde_steps}~{num_inference_steps-1}: ODE)
    Num train timesteps: {num_train_timesteps} (only SDE steps are trained)
    Guidance scale: {guidance_scale}
    Sample batch size per GPU: {sample_batch_size}
    Train batch size per GPU: {train_batch_size}
    Num batches per epoch: {num_batches_per_epoch}
    Num inner epochs: {num_inner_epochs}
    Num image per prompt: {num_image_per_prompt}
    Clip range: {clip_range}
    Adv clip max: {adv_clip_max}
    KL reward: {kl_reward}
    KL beta: {kl_beta}
    GT teacher forcing: {use_gt_teacher_forcing}
    GT teacher loss weight: {gt_teacher_loss_weight}
    GT teacher positive only: {gt_teacher_positive_only}
    GT teacher train batch size: {gt_teacher_train_batch_size}
    GT teacher every n steps: {gt_teacher_every_n_steps}
    Off-policy guidance: {use_offpolicy_guidance}
    Off-policy trace dir: {offpolicy_trace_dir}
    Off-policy loss weight: {offpolicy_loss_weight}
    Off-policy surrogate: {offpolicy_surrogate}
    Off-policy advantage mode: {offpolicy_advantage_mode}
    Off-policy train batch size: {offpolicy_train_batch_size}
    Off-policy every n steps: {offpolicy_every_n_steps}
    Off-policy max steps: {offpolicy_max_steps}
    Off-policy step stride: {offpolicy_step_stride}
    Off-policy step weighting: {offpolicy_step_weighting}
    Off-policy use teacher logprob: {offpolicy_use_teacher_logprob}
    Debug training trace: {debug_training}
    Debug trace scope: epochs={debug_training_epochs}, inner_epochs={debug_training_inner_epochs}, micro_batches={debug_training_micro_batches}, timesteps={debug_training_timesteps}, sync={debug_training_sync}, stack_timeout={debug_stack_timeout}s
    Per-prompt stat tracking: {per_prompt_stat_tracking}
    Gradient checkpointing: {gradient_checkpointing}
    Weight dtype: {weight_dtype}
    EMA decay: {ema_decay}
    Learning rate: {learning_rate}
    Resume global step: {resume_global_step}
    Start epoch: {start_epoch}
    Resume epoch override: {resume_epoch}
    Resume LR override: {resume_override_lr}
    Resume LR value: {resume_lr if resume_lr is not None else 'follow optimizer_config.lr'}
    Current optimizer lr: {optimizer.param_groups[0]['lr']}
    Gradient accumulation steps: {gradient_accumulation_steps}
    FSDP size: {fsdp_size}
    DDP size: {ddp_size}
    World size: {world_size}
    dp_size: {dp_size}
    sp_size: {sp_size}
    skiparse_sp_size: {skiparse_sp_size}
    global_sp_size: {global_sp_size}
    Use Sequence Parallel: {use_sequence_parallel}
    Use Skiparse Sequence Parallel: {use_skiparse_sequence_parallel}
    Use Full Blocks Sequence Parallel: {use_full_blocks_sequence_parallel}
    Reshard after forward: {reshard_after_forward}
    Model CPU offload: {model_cpu_offload}
    Video: {video_num_frames}f x {video_height}h x {video_width}w
    Output dir: {output_dir}
    {'=' * 20}{'=' * len('Start RL Training (GRPO + LoRA + FSDP2)')}{'=' * 20}
    """)

    global_step = resume_global_step
    last_completed_epoch = start_epoch - 1
    train_iter = iter(train_dataloader)

    vae_temporal_factor = 4
    vae_spatial_factor = 8
    latent_T = (video_num_frames - 1) // vae_temporal_factor + 1
    latent_H = video_height // vae_spatial_factor
    latent_W = video_width // vae_spatial_factor
    latent_C = model_config.get("in_dim", 16)
    latent_shape = (sample_batch_size, latent_C, latent_T, latent_H, latent_W)

    log_on_main_process(logger, f"Latent shape: {latent_shape}")

    accum_steps_total = gradient_accumulation_steps * num_train_timesteps

    for epoch in range(start_epoch, num_epochs):
        if offpolicy_sampler is not None:
            offpolicy_sampler.set_epoch(epoch)
            offpolicy_iterator = None

        ### — — — — — — Sample — — — — — — 
        model.eval()

        if reshard_after_forward is not None and not reshard_after_forward:
            model.set_reshard_after_forward(True, recurse=True)

        samples = []
        all_group_keys = []
        gt_samples = []  # off-policy GT reward futures (do NOT participate in training)
        gt_teacher_samples = []  # local GT latents used by the stage-2 teacher-forcing branch
        gt_seen_group_keys = set()
        last_videos_cpu = None
        last_prompts = None

        for batch_idx in tqdm(
            range(num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=(rank != 0),
        ):
            train_sampler.set_epoch(epoch * num_batches_per_epoch + batch_idx)
            batch = next(train_iter)
            prompts = batch[PROMPT]
            prompt_ids = batch[PROMPT_IDS].to(device)
            prompt_mask = batch[PROMPT_MASK].to(device)
            prompt_metadata = batch["metadata"]
            group_keys = batch.get("group_key", prompts)
            unique_gt_indices = []
            gt_group_keys = []
            teacher_gt_indices = []
            teacher_gt_group_keys = []

            # Off-policy GT guidance: score the ground-truth video for this batch.
            # The reward joins the GRPO group (advantage baseline) but the GT
            # trajectory itself never enters `samples`, so it cannot affect the
            # gradient / parameter update.
            if use_gt_guidance and VIDEO in batch and batch[VIDEO] is not None:
                if use_gt_teacher_forcing:
                    teacher_gt_indices = list(range(len(group_keys)))
                    teacher_gt_group_keys = list(group_keys)

                for local_idx, group_key in enumerate(group_keys):
                    if group_key in gt_seen_group_keys:
                        continue
                    gt_seen_group_keys.add(group_key)
                    unique_gt_indices.append(local_idx)

                if unique_gt_indices:
                    gt_videos_cpu = batch[VIDEO][unique_gt_indices].detach().float().cpu()  # (B, C, T, H, W) in [-1, 1]
                    gt_videos_for_reward = (gt_videos_cpu + 1.0) / 2.0
                    gt_prompts = [prompts[i] for i in unique_gt_indices]
                    gt_metadata = [prompt_metadata[i] for i in unique_gt_indices]
                    gt_group_keys = [group_keys[i] for i in unique_gt_indices]
                    gt_rewards_future = executor.submit(
                        reward_fn, gt_videos_for_reward.numpy(), gt_prompts, gt_metadata, True
                    )
                    gt_samples.append({
                        "group_keys": gt_group_keys,
                        "rewards": gt_rewards_future,
                    })
                    del gt_videos_cpu, gt_videos_for_reward

            if encoder_cpu_offload:
                vae.model.to(device)
                if not text_encoder_use_fsdp:
                    text_encoder.model.to(device)

            with torch.no_grad():
                text_embeddings = text_encoder(prompt_ids, prompt_mask)
            torch.cuda.synchronize()

            if use_gt_teacher_forcing and teacher_gt_indices:
                with torch.no_grad():
                    gt_videos_for_teacher = batch[VIDEO][teacher_gt_indices].to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    gt_latents = vae.encode(gt_videos_for_teacher)
                gt_teacher_samples.append({
                    "group_keys": list(teacher_gt_group_keys),
                    "latents": gt_latents.detach().cpu(),
                    "prompt_embeds": text_embeddings[teacher_gt_indices].detach().cpu(),
                })
                del gt_videos_for_teacher, gt_latents

            if encoder_cpu_offload:
                vae.model.to("cpu")
                if not text_encoder_use_fsdp:
                    text_encoder.model.to("cpu")
                torch.cuda.empty_cache()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            for sample_t in range(sample_time_per_prompt):
                all_group_keys.extend(group_keys)
                with torch.no_grad():
                    videos, latents_list, log_probs_list, kl_list = osp_sample_with_logprob(
                        model=model,
                        scheduler=scheduler,
                        vae=vae,
                        latent_shape=latent_shape,
                        text_embeddings=text_embeddings,
                        device=device,
                        weight_dtype=weight_dtype,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_text_embeddings=neg_text_embeddings.expand(sample_batch_size, -1, -1),
                        start_frame_latents=None,
                        determistic=False,
                        kl_reward=kl_reward,
                        sp_group=global_sp_group,
                        sde_steps=sde_steps,
                    )

                latents_stacked = torch.stack(latents_list, dim=1)
                log_probs_stacked = torch.stack(log_probs_list, dim=1)
                kl_stacked = torch.stack(kl_list, dim=1)
                del latents_list, log_probs_list, kl_list

                timesteps_repeated = torch.arange(sde_steps, device=device).unsqueeze(0).expand(
                    sample_batch_size, -1
                )

                videos_cpu = videos.detach().cpu()
                del videos
                videos_for_reward = (videos_cpu.float() + 1.0) / 2.0
                rewards_future = executor.submit(
                    reward_fn, videos_for_reward.numpy(), prompts, prompt_metadata, True
                )
                del videos_for_reward

                last_videos_cpu = videos_cpu
                last_prompts = list(prompts)

                samples.append({
                    "prompt_embeds": text_embeddings.detach().cpu(),
                    "neg_prompt_embeds": neg_text_embeddings.expand(sample_batch_size, -1, -1).detach().cpu(),
                    "timesteps": timesteps_repeated.cpu(),
                    "latents": latents_stacked[:, :-1].detach(),
                    "next_latents": latents_stacked[:, 1:].detach(),
                    "log_probs": log_probs_stacked.detach(),
                    "kl": kl_stacked.detach(),
                    "rewards": rewards_future,
                })
                del latents_stacked, log_probs_stacked, kl_stacked, videos_cpu
                torch.cuda.empty_cache()

            del text_embeddings, prompt_ids, prompt_mask

        for sample in tqdm(samples, desc="Waiting for rewards", disable=(rank != 0)):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=device).float()
                for key, value in rewards.items()
            }

        # Resolve the off-policy GT rewards. We keep only the local-rank GT
        # group keys and their `avg` reward; they will be gathered and folded into
        # the GRPO group statistics below.
        gt_local_group_keys = []
        gt_local_avg_rewards = []
        if use_gt_guidance:
            for gt_sample in tqdm(gt_samples, desc="Waiting for GT rewards", disable=(rank != 0)):
                gt_rewards, _ = gt_sample["rewards"].result()
                gt_avg = np.asarray(gt_rewards["avg"], dtype=np.float64).reshape(-1)
                if len(gt_avg) != len(gt_sample["group_keys"]):
                    raise RuntimeError(
                        "GT reward count does not match GT group key count: "
                        f"{len(gt_avg)} rewards vs {len(gt_sample['group_keys'])} keys"
                    )
                gt_local_group_keys.extend(gt_sample["group_keys"])
                gt_local_avg_rewards.extend(gt_avg.tolist())

        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if rank == 0 and last_videos_cpu is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_vis = min(8, len(last_videos_cpu))
                sample_indices = random.sample(range(len(last_videos_cpu)), num_vis)
                for idx, i in enumerate(sample_indices):
                    video = last_videos_cpu[i].numpy().transpose(1, 2, 3, 0)
                    frames = [((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8) for frame in video]
                    imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=16, codec="libx264", format='FFMPEG')

                if wandb.run is not None:
                    sampled_prompts = [last_prompts[i] for i in sample_indices]
                    wandb.log(
                        {"videos": [
                            wandb.Video(os.path.join(tmpdir, f"{idx}.mp4"), caption=f"{p:.100}", fps=16)
                            for idx, p in enumerate(sampled_prompts)
                        ]},
                        step=global_step,
                    )

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        kl_on_device = samples["kl"].to(device)
        num_steps_dim = kl_on_device.shape[1] if kl_on_device.dim() > 1 else sde_steps
        avg_expanded = samples["rewards"]["avg"].unsqueeze(-1).expand(-1, num_steps_dim)
        samples["rewards"]["avg"] = avg_expanded - kl_reward * kl_on_device
        del kl_on_device

        gathered_rewards = {}
        for key, value in samples["rewards"].items():
            if value.dim() == 1:
                gathered = gather_data_from_all_ranks(value.unsqueeze(0), dim=0, group=dp_group if use_global_sequence_parallel else None)
                gathered_rewards[key] = gathered.reshape(-1).cpu().numpy()
            else:
                gathered = gather_data_from_all_ranks(value, dim=0, group=dp_group if use_global_sequence_parallel else None)
                gathered_rewards[key] = gathered.reshape(-1, *value.shape[1:]).cpu().numpy()

        if rank == 0:
            log_dict = {
                "epoch": epoch,
                "kl": samples["kl"].mean().cpu().item(),
                "kl_abs": samples["kl"].abs().mean().cpu().item(),
            }

            for key, value in gathered_rewards.items():
                if '_strict_accuracy' not in key and '_accuracy' not in key:
                    log_dict[f"reward_{key}"] = float(value.mean())
                    log_dict[f"reward/{key}_mean"] = float(value.mean())
                    log_dict[f"reward/{key}_std"] = float(value.std())
                    log_dict[f"reward/{key}_abs_mean"] = float(np.abs(value).mean())
                    log_dict[f"reward/{key}_max"] = float(value.max())
                    log_dict[f"reward/{key}_min"] = float(value.min())

            if wandb.run is not None:
                wandb.log(log_dict, step=global_step)

        # Gather the off-policy GT (ground-truth) group keys and their `avg` rewards
        # from every dp rank. These reference rewards are folded into the GRPO
        # group statistics (to shift the advantage baseline) but never produce
        # advantages of their own, so they cannot drive gradient updates.
        gt_global_group_keys = None
        gt_global_rewards = None
        gt_global_advantages = None
        if use_gt_guidance:
            gathered_gt_group_keys_list = [None] * dp_size
            gathered_gt_rewards_list = [None] * dp_size
            if use_global_sequence_parallel:
                dist.all_gather_object(gathered_gt_group_keys_list, gt_local_group_keys, group=dp_group)
                dist.all_gather_object(gathered_gt_rewards_list, gt_local_avg_rewards, group=dp_group)
            else:
                dist.all_gather_object(gathered_gt_group_keys_list, gt_local_group_keys)
                dist.all_gather_object(gathered_gt_rewards_list, gt_local_avg_rewards)

            dedup_gt_rewards = {}
            for rank_group_keys, rank_rewards in zip(gathered_gt_group_keys_list, gathered_gt_rewards_list):
                for group_key, reward in zip(rank_group_keys, rank_rewards):
                    dedup_gt_rewards.setdefault(group_key, reward)

            gt_global_group_keys = list(dedup_gt_rewards.keys())
            gt_global_rewards = list(dedup_gt_rewards.values())
            if len(gt_global_group_keys) == 0:
                gt_global_group_keys = None
                gt_global_rewards = None
            elif rank == 0:
                gt_arr = np.asarray(gt_global_rewards, dtype=np.float64)
                log_on_main_process(
                    logger,
                    f"Epoch {epoch} | GT guidance: {len(gt_global_group_keys)} ref videos | "
                    f"gt_reward mean: {gt_arr.mean():.4f} | std: {gt_arr.std():.4f}",
                )
                if wandb.run is not None:
                    wandb.log({
                        "gt_guidance/reward_mean": float(gt_arr.mean()),
                        "gt_guidance/reward_std": float(gt_arr.std()),
                        "gt_guidance/num_ref": len(gt_global_group_keys),
                    }, step=global_step)

        if per_prompt_stat_tracking and stat_tracker is not None:
            gathered_group_keys_list = [None] * dp_size
            if use_global_sequence_parallel:
                dist.all_gather_object(gathered_group_keys_list, all_group_keys, group=dp_group)
            else:
                dist.all_gather_object(gathered_group_keys_list, all_group_keys)
            gathered_group_keys = [p for rank_group_keys in gathered_group_keys_list for p in rank_group_keys]
            if len(gathered_group_keys) != len(gathered_rewards['ori_avg']):
                raise RuntimeError(
                    "Rollout group key count does not match rollout reward count: "
                    f"{len(gathered_group_keys)} keys vs {len(gathered_rewards['ori_avg'])} rewards"
                )
            if use_gt_teacher_forcing and gt_global_group_keys is not None:
                advantages, gt_global_advantages = stat_tracker.update(
                    gathered_group_keys,
                    gathered_rewards['ori_avg'],
                    ref_prompts=gt_global_group_keys,
                    ref_rewards=gt_global_rewards,
                    return_ref_advantages=True,
                )
            else:
                advantages = stat_tracker.update(
                    gathered_group_keys,
                    gathered_rewards['ori_avg'],
                    ref_prompts=gt_global_group_keys,
                    ref_rewards=gt_global_rewards,
                )

            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio = calculate_zero_std_ratio(gathered_group_keys, gathered_rewards)
            if rank == 0 and wandb.run is not None:
                wandb.log({
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                }, step=global_step)
            stat_tracker.clear()
        else:
            avg_rewards = gathered_rewards['ori_avg']
            # Fold GT rewards into the global baseline mean/std (off-policy guidance),
            # without emitting advantages for the GT samples themselves.
            if gt_global_rewards is not None and len(gt_global_rewards) > 0:
                baseline_source = np.concatenate(
                    [avg_rewards, np.asarray(gt_global_rewards, dtype=avg_rewards.dtype)], axis=0
                )
                advantages = (avg_rewards - baseline_source.mean()) / (baseline_source.std() + 1e-4)
                if use_gt_teacher_forcing:
                    gt_rewards_arr = np.asarray(gt_global_rewards, dtype=avg_rewards.dtype)
                    gt_global_advantages = (gt_rewards_arr - baseline_source.mean()) / (baseline_source.std() + 1e-4)
            else:
                advantages = (avg_rewards - avg_rewards.mean()) / (avg_rewards.std() + 1e-4)

        advantages = torch.as_tensor(advantages).float()

        if rank == 0:
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item()
            adv_abs_mean = advantages.abs().mean().item()
            adv_max = advantages.max().item()
            adv_min = advantages.min().item()
            log_on_main_process(logger, f"Epoch {epoch} | advantage mean: {adv_mean:.4f} | advantage std: {adv_std:.4f} | advantage abs mean: {adv_abs_mean:.4f} | advantage max: {adv_max:.4f} | advantage min: {adv_min:.4f}")
            if wandb.run is not None:
                wandb.log({
                    "advantage/mean": adv_mean,
                    "advantage/std": adv_std,
                    "advantage/abs_mean": adv_abs_mean,
                    "advantage/max": adv_max,
                    "advantage/min": adv_min,
                }, step=global_step)

        if advantages.dim() == 1:
            local_advantages = advantages.reshape(dp_size, -1)[dp_rank]
            local_advantages = local_advantages.unsqueeze(-1).expand(-1, num_train_timesteps).contiguous()
        else:
            local_advantages = advantages.reshape(dp_size, -1, *advantages.shape[1:])[dp_rank]
        samples["advantages"] = local_advantages

        gt_teacher_data = None
        if use_gt_teacher_forcing and gt_teacher_samples and gt_global_group_keys is not None and gt_global_advantages is not None:
            gt_advantage_by_key = {
                group_key: adv
                for group_key, adv in zip(gt_global_group_keys, np.asarray(gt_global_advantages, dtype=np.float32))
            }
            prepared_gt_teacher_samples = []
            for gt_teacher_sample in gt_teacher_samples:
                local_gt_advantages = [
                    gt_advantage_by_key[group_key]
                    for group_key in gt_teacher_sample["group_keys"]
                    if group_key in gt_advantage_by_key
                ]
                if len(local_gt_advantages) != len(gt_teacher_sample["group_keys"]):
                    continue
                prepared_gt_teacher_samples.append({
                    "latents": gt_teacher_sample["latents"],
                    "prompt_embeds": gt_teacher_sample["prompt_embeds"],
                    "advantages": torch.as_tensor(local_gt_advantages, dtype=torch.float32),
                })
            if prepared_gt_teacher_samples:
                gt_teacher_data = {
                    "latents": torch.cat([s["latents"] for s in prepared_gt_teacher_samples], dim=0),
                    "prompt_embeds": torch.cat([s["prompt_embeds"] for s in prepared_gt_teacher_samples], dim=0),
                    "advantages": torch.cat([s["advantages"] for s in prepared_gt_teacher_samples], dim=0),
                }

        if use_gt_teacher_forcing:
            local_teacher_count = len(gt_teacher_data["latents"]) if gt_teacher_data is not None else 0
            min_teacher_count = torch.tensor(local_teacher_count, device=device, dtype=torch.long)
            max_teacher_count = torch.tensor(local_teacher_count, device=device, dtype=torch.long)
            dist.all_reduce(min_teacher_count, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_teacher_count, op=dist.ReduceOp.MAX)
            synced_teacher_count = int(min_teacher_count.item())
            if rank == 0 and int(max_teacher_count.item()) != synced_teacher_count:
                log_on_main_process(
                    logger,
                    "GT teacher sample count differs across ranks; "
                    f"truncate to global min {synced_teacher_count} "
                    f"(max {int(max_teacher_count.item())}).",
                )
            if synced_teacher_count == 0:
                gt_teacher_data = None
            elif gt_teacher_data is not None:
                gt_teacher_data = {
                    k: v[:synced_teacher_count]
                    for k, v in gt_teacher_data.items()
                }
                if rank == 0:
                    log_on_main_process(
                        logger,
                        f"Epoch {epoch} | GT teacher samples: {len(gt_teacher_data['latents'])} | "
                        f"adv mean: {gt_teacher_data['advantages'].mean().item():.4f}",
                    )

        if rank == 0:
            log_on_main_process(logger, f"Epoch {epoch} | local advantages abs mean: {samples['advantages'].abs().mean().item():.4f} | kl mean: {samples['kl'].mean().item():.4f}")

        del samples["rewards"]

        mask = (samples["advantages"].abs().sum(dim=1) != 0) if samples["advantages"].dim() > 1 else (samples["advantages"].abs() != 0)

        num_batches_total = num_batches_per_epoch * sample_time_per_prompt
        true_count = mask.sum()
        if true_count == 0:
            samples["advantages"] = samples["advantages"] + 1e-6
            mask = torch.ones(len(samples["advantages"]), dtype=torch.bool)

        if true_count % num_batches_total != 0 and true_count > 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches_total - (true_count % num_batches_total)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True

        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size_local = len(samples["timesteps"])
        min_train_count = torch.tensor(total_batch_size_local, device=device, dtype=torch.long)
        max_train_count = torch.tensor(total_batch_size_local, device=device, dtype=torch.long)
        dist.all_reduce(min_train_count, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_train_count, op=dist.ReduceOp.MAX)
        synced_train_count = int(min_train_count.item())
        if synced_train_count <= 0:
            raise RuntimeError(
                "No rollout samples remain after advantage masking on at least one rank."
            )
        if int(max_train_count.item()) != synced_train_count and rank == 0:
            log_on_main_process(
                logger,
                "Rollout sample count differs across ranks after advantage masking; "
                f"truncate to global min {synced_train_count} "
                f"(max {int(max_train_count.item())}).",
            )
        if total_batch_size_local != synced_train_count:
            samples = {
                k: (
                    {sub_k: sub_v[:synced_train_count] for sub_k, sub_v in v.items()}
                    if isinstance(v, dict)
                    else v[:synced_train_count]
                )
                for k, v in samples.items()
            }
            total_batch_size_local = synced_train_count

        backward_counter = 0

        for inner_epoch in range(num_inner_epochs):
            model.train()
            if reshard_after_forward is not None and not reshard_after_forward:
                model.set_reshard_after_forward(reshard_after_forward, recurse=True)

            debug_train_log(
                "inner_start",
                epoch,
                inner_epoch,
                extra=(
                    f"total_batch_size_local={total_batch_size_local}, "
                    f"train_batch_size={train_batch_size}, "
                    f"gt_teacher_count={len(gt_teacher_data['latents']) if gt_teacher_data is not None else 0}"
                ),
            )
            debug_train_log("perm_randperm_before", epoch, inner_epoch)
            perm = torch.randperm(total_batch_size_local, device=device)
            debug_train_log("perm_randperm_after", epoch, inner_epoch, sync=debug_training_sync)
            if use_global_sequence_parallel:
                debug_train_log("perm_broadcast_before", epoch, inner_epoch)
                torch.distributed.broadcast(perm, group_src=0, group=global_sp_group)
                debug_train_log("perm_broadcast_after", epoch, inner_epoch, sync=debug_training_sync)
            debug_train_log("perm_cpu_before", epoch, inner_epoch)
            perm = perm.cpu()
            debug_train_log("perm_cpu_after", epoch, inner_epoch)

            debug_train_log("sample_shuffle_before", epoch, inner_epoch)
            samples = {
                k: (
                    {sub_k: sub_v[perm] for sub_k, sub_v in v.items()}
                    if isinstance(v, dict)
                    else v[perm]
                )
                for k, v in samples.items()
            }
            debug_train_log("sample_shuffle_after", epoch, inner_epoch)

            gt_teacher_epoch_data = None
            if gt_teacher_data is not None:
                gt_teacher_total = len(gt_teacher_data["latents"])
                debug_train_log("gt_perm_randperm_before", epoch, inner_epoch, extra=f"gt_teacher_total={gt_teacher_total}")
                gt_perm = torch.randperm(gt_teacher_total, device=device)
                debug_train_log("gt_perm_randperm_after", epoch, inner_epoch, sync=debug_training_sync)
                if use_global_sequence_parallel:
                    debug_train_log("gt_perm_broadcast_before", epoch, inner_epoch)
                    torch.distributed.broadcast(gt_perm, group_src=0, group=global_sp_group)
                    debug_train_log("gt_perm_broadcast_after", epoch, inner_epoch, sync=debug_training_sync)
                debug_train_log("gt_perm_cpu_before", epoch, inner_epoch)
                gt_perm = gt_perm.cpu()
                debug_train_log("gt_perm_cpu_after", epoch, inner_epoch)
                debug_train_log("gt_teacher_shuffle_before", epoch, inner_epoch)
                gt_teacher_epoch_data = {k: v[gt_perm] for k, v in gt_teacher_data.items()}
                debug_train_log("gt_teacher_shuffle_after", epoch, inner_epoch)

            num_micro_batches = max(1, total_batch_size_local // train_batch_size)
            debug_train_log("micro_batches_ready", epoch, inner_epoch, extra=f"num_micro_batches={num_micro_batches}")

            info = defaultdict(list)
            gt_teacher_cursor = 0
            for mb_idx in tqdm(
                range(num_micro_batches),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                disable=(rank != 0),
            ):
                mb_start = mb_idx * train_batch_size
                mb_end = min(mb_start + train_batch_size, total_batch_size_local)
                debug_train_log(
                    "micro_batch_to_device_before",
                    epoch,
                    inner_epoch,
                    mb_idx,
                    extra=f"range={mb_start}:{mb_end}",
                )
                micro_batch = {
                    k: (
                        {sub_k: sub_v[mb_start:mb_end].to(device) for sub_k, sub_v in v.items()}
                        if isinstance(v, dict)
                        else v[mb_start:mb_end].to(device)
                    )
                    for k, v in samples.items()
                }
                debug_train_log(
                    "micro_batch_to_device_after",
                    epoch,
                    inner_epoch,
                    mb_idx,
                    extra=f"batch={len(micro_batch['timesteps'])}",
                    sync=debug_training_sync,
                )

                embeds = micro_batch["prompt_embeds"]
                neg_embeds = micro_batch["neg_prompt_embeds"] if use_cfg_in_train else None

                for j in train_timestep_indices:
                    debug_train_log("timestep_start", epoch, inner_epoch, mb_idx, j)
                    debug_train_log("pre_forward_sync_before", epoch, inner_epoch, mb_idx, j)
                    torch.cuda.synchronize()
                    debug_train_log("pre_forward_sync_after", epoch, inner_epoch, mb_idx, j)
                    debug_train_log("policy_forward_before", epoch, inner_epoch, mb_idx, j)
                    with torch.autocast("cuda", dtype=weight_dtype):
                        prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = compute_log_prob_for_training(
                            model=model,
                            sample=micro_batch,
                            step_idx=j,
                            text_embeddings=embeds,
                            weight_dtype=weight_dtype,
                            sigmas_schedule=sigmas_schedule,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale if use_cfg_in_train else 1.0,
                            negative_text_embeddings=neg_embeds,
                            start_frame_latents=None,
                            sp_group=global_sp_group,
                        )
                    debug_train_log("policy_forward_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)

                    if kl_beta > 0:
                        debug_train_log("ref_forward_before", epoch, inner_epoch, mb_idx, j)
                        with torch.no_grad():
                            with model.disable_adapter():
                                with torch.autocast("cuda", dtype=weight_dtype):
                                    _, _, ref_prev_sample_mean, ref_std_dev_t, ref_dt = compute_log_prob_for_training(
                                        model=model,
                                        sample=micro_batch,
                                        step_idx=j,
                                        text_embeddings=embeds,
                                        weight_dtype=weight_dtype,
                                        sigmas_schedule=sigmas_schedule,
                                        num_inference_steps=num_inference_steps,
                                        guidance_scale=guidance_scale if use_cfg_in_train else 1.0,
                                        negative_text_embeddings=neg_embeds,
                                        start_frame_latents=None,
                                        sp_group=global_sp_group,
                                    )
                        debug_train_log("ref_forward_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)

                    if micro_batch["advantages"].dim() > 1:
                        adv = torch.clamp(micro_batch["advantages"][:, j], -adv_clip_max, adv_clip_max)
                    else:
                        adv = torch.clamp(micro_batch["advantages"], -adv_clip_max, adv_clip_max)

                    ratio = torch.exp(log_prob - micro_batch["log_probs"][:, j])
                    unclipped_loss = -adv * ratio
                    clipped_loss = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                    if kl_beta > 0:
                        kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(dim=(1, 2, 3, 4)) / (2 * (std_dev_t * ref_dt) ** 2).squeeze()
                        kl_loss = torch.mean(kl_loss)
                        loss = policy_loss + kl_beta * kl_loss
                        info["kl_loss"].append(kl_loss.detach())
                    else:
                        loss = policy_loss

                    approx_kl = 0.5 * torch.mean((log_prob - micro_batch["log_probs"][:, j]) ** 2).detach()
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach()
                    info["policy_loss"].append(policy_loss.detach())

                    loss = loss / accum_steps_total
                    debug_train_log("policy_backward_before", epoch, inner_epoch, mb_idx, j)
                    loss.backward()
                    debug_train_log("policy_backward_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)
                    total_loss_for_log = loss.detach()

                    info["approx_kl"].append(approx_kl)
                    info["clipfrac"].append(clipfrac)

                    # Backprop the policy/kl graph before the GT teacher-forcing
                    # forward so their model activations do not overlap in memory.
                    del prev_sample, log_prob, prev_sample_mean, std_dev_t, dt
                    del ratio, unclipped_loss, clipped_loss, policy_loss, loss
                    if kl_beta > 0:
                        del ref_prev_sample_mean, ref_std_dev_t, ref_dt, kl_loss

                    gt_teacher_loss = None
                    gt_teacher_metrics = None
                    if gt_teacher_epoch_data is not None and (j % gt_teacher_every_n_steps == 0):
                        gt_teacher_total = len(gt_teacher_epoch_data["latents"])
                        gt_indices = (torch.arange(gt_teacher_train_batch_size) + gt_teacher_cursor) % gt_teacher_total
                        gt_teacher_cursor += gt_teacher_train_batch_size
                        debug_train_log(
                            "gt_teacher_to_device_before",
                            epoch,
                            inner_epoch,
                            mb_idx,
                            j,
                            extra=f"gt_teacher_total={gt_teacher_total}, gt_batch={gt_teacher_train_batch_size}",
                        )
                        gt_teacher_micro_batch = {
                            k: v[gt_indices].to(device)
                            for k, v in gt_teacher_epoch_data.items()
                        }
                        debug_train_log("gt_teacher_to_device_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)
                        debug_train_log("gt_teacher_forward_before", epoch, inner_epoch, mb_idx, j)
                        with torch.autocast("cuda", dtype=weight_dtype):
                            gt_teacher_loss, gt_teacher_metrics = compute_gt_teacher_forcing_loss(
                                model=model,
                                scheduler=scheduler,
                                gt_latents=gt_teacher_micro_batch["latents"],
                                text_embeddings=gt_teacher_micro_batch["prompt_embeds"],
                                gt_advantages=gt_teacher_micro_batch["advantages"],
                                weight_dtype=weight_dtype,
                                loss_weight=gt_teacher_loss_weight,
                                adv_clip_max=gt_teacher_adv_clip_max,
                                positive_only=gt_teacher_positive_only,
                                sp_group=global_sp_group,
                            )
                        debug_train_log("gt_teacher_forward_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)

                        info["gt_teacher_loss"].append(gt_teacher_loss.detach())
                        if gt_teacher_metrics is not None:
                            info["gt_teacher_active_ratio"].append(
                                torch.tensor(gt_teacher_metrics["gt_teacher/active_ratio"], device=device)
                            )
                            info["gt_teacher_adv_mean"].append(
                                torch.tensor(gt_teacher_metrics["gt_teacher/adv_mean"], device=device)
                            )
                        if gt_teacher_loss.requires_grad:
                            gt_teacher_loss = gt_teacher_loss / accum_steps_total
                            debug_train_log("gt_teacher_backward_before", epoch, inner_epoch, mb_idx, j)
                            gt_teacher_loss.backward()
                            debug_train_log("gt_teacher_backward_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)
                            total_loss_for_log = total_loss_for_log + gt_teacher_loss.detach()
                        del gt_teacher_micro_batch, gt_teacher_loss

                    offpolicy_loss = None
                    offpolicy_metrics = None
                    if offpolicy_dataloader is not None and (j % offpolicy_every_n_steps == 0):
                        debug_train_log("offpolicy_batch_fetch_before", epoch, inner_epoch, mb_idx, j)
                        offpolicy_cpu_batch = next_offpolicy_batch()
                        debug_train_log("offpolicy_batch_fetch_after", epoch, inner_epoch, mb_idx, j)
                        debug_train_log(
                            "offpolicy_encode_before",
                            epoch,
                            inner_epoch,
                            mb_idx,
                            j,
                            extra=f"offpolicy_batch={len(offpolicy_cpu_batch[PROMPT])}",
                        )
                        offpolicy_text_embeddings = encode_text_batch_for_loss(
                            offpolicy_cpu_batch[PROMPT_IDS],
                            offpolicy_cpu_batch[PROMPT_MASK],
                        )
                        debug_train_log("offpolicy_encode_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)

                        offpolicy_micro_batch = {
                            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                            for key, value in offpolicy_cpu_batch.items()
                        }
                        debug_train_log("offpolicy_forward_before", epoch, inner_epoch, mb_idx, j)
                        with torch.autocast("cuda", dtype=weight_dtype):
                            offpolicy_loss, offpolicy_metrics = compute_offpolicy_guidance_loss(
                                model=model,
                                trace_batch=offpolicy_micro_batch,
                                text_embeddings=offpolicy_text_embeddings,
                                weight_dtype=weight_dtype,
                                sigmas_schedule=sigmas_schedule,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale if use_cfg_in_train else 1.0,
                                negative_text_embeddings=neg_text_embeddings if use_cfg_in_train else None,
                                loss_weight=offpolicy_loss_weight,
                                adv_clip_max=offpolicy_adv_clip_max,
                                ratio_clip=offpolicy_ratio_clip,
                                positive_only=offpolicy_positive_only,
                                use_teacher_logprob=offpolicy_use_teacher_logprob,
                                surrogate=offpolicy_surrogate,
                                step_weighting=offpolicy_step_weighting,
                                step_weight_gamma=offpolicy_step_weight_gamma,
                                sp_group=global_sp_group,
                            )
                        debug_train_log("offpolicy_forward_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)

                        info["offpolicy_loss"].append(offpolicy_loss.detach())
                        if offpolicy_metrics is not None:
                            info["offpolicy_active_ratio"].append(
                                torch.tensor(offpolicy_metrics["offpolicy/active_ratio"], device=device)
                            )
                            info["offpolicy_ratio_mean"].append(
                                torch.tensor(offpolicy_metrics["offpolicy/ratio_mean"], device=device)
                            )
                            info["offpolicy_adv_mean"].append(
                                torch.tensor(offpolicy_metrics["offpolicy/adv_mean"], device=device)
                            )
                            info["offpolicy_logprob_mean"].append(
                                torch.tensor(offpolicy_metrics["offpolicy/logprob_mean"], device=device)
                            )
                        if offpolicy_loss.requires_grad:
                            offpolicy_loss = offpolicy_loss / accum_steps_total
                            debug_train_log("offpolicy_backward_before", epoch, inner_epoch, mb_idx, j)
                            offpolicy_loss.backward()
                            debug_train_log("offpolicy_backward_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)
                            total_loss_for_log = total_loss_for_log + offpolicy_loss.detach()
                        del offpolicy_cpu_batch, offpolicy_micro_batch, offpolicy_text_embeddings, offpolicy_loss

                    info["loss"].append(total_loss_for_log)

                    backward_counter += 1

                    if backward_counter % accum_steps_total == 0:
                        debug_train_log("grad_clip_before", epoch, inner_epoch, mb_idx, j)
                        grad_norm = adaptive_grad_clipper.adaptive_clip(trainable_parameters)
                        debug_train_log("grad_clip_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)
                        debug_train_log("optimizer_step_before", epoch, inner_epoch, mb_idx, j)
                        optimizer.step()
                        debug_train_log("optimizer_step_after", epoch, inner_epoch, mb_idx, j, sync=debug_training_sync)
                        debug_train_log("zero_grad_before", epoch, inner_epoch, mb_idx, j)
                        model.zero_grad(set_to_none=True)
                        debug_train_log("zero_grad_after", epoch, inner_epoch, mb_idx, j)
                        debug_train_log("ema_update_before", epoch, inner_epoch, mb_idx, j)
                        ema_model.update(model, global_step + 1)
                        debug_train_log("ema_update_after", epoch, inner_epoch, mb_idx, j)
                        global_step += 1

                        if len(info) > 0:
                            info_mean = {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}
                            if rank == 0:
                                tqdm.write(
                                    f"  step {global_step} | loss: {info_mean.get('loss', 0):.6f} | "
                                    f"policy_loss: {info_mean.get('policy_loss', 0):.6f} | "
                                    f"gt_teacher_loss: {info_mean.get('gt_teacher_loss', 0):.6f} | "
                                    f"offpolicy_loss: {info_mean.get('offpolicy_loss', 0):.6f} | "
                                    f"approx_kl: {info_mean.get('approx_kl', 0):.6f} | "
                                    f"clipfrac: {info_mean.get('clipfrac', 0):.4f} | "
                                    f"grad_norm: {grad_norm.item():.4f}"
                                )
                                if wandb.run is not None:
                                    wandb_log = {
                                        "train/loss": info_mean.get("loss", 0),
                                        "train/policy_loss": info_mean.get("policy_loss", 0),
                                        "train/approx_kl": info_mean.get("approx_kl", 0),
                                        "train/clipfrac": info_mean.get("clipfrac", 0),
                                        "train/grad_norm": grad_norm.item(),
                                        "train/lr": optimizer.param_groups[0]['lr'],
                                    }
                                    if "kl_loss" in info_mean:
                                        wandb_log["train/kl_loss"] = info_mean["kl_loss"]
                                    if "gt_teacher_loss" in info_mean:
                                        wandb_log["train/gt_teacher_loss"] = info_mean["gt_teacher_loss"]
                                    if "gt_teacher_active_ratio" in info_mean:
                                        wandb_log["train/gt_teacher_active_ratio"] = info_mean["gt_teacher_active_ratio"]
                                    if "gt_teacher_adv_mean" in info_mean:
                                        wandb_log["train/gt_teacher_adv_mean"] = info_mean["gt_teacher_adv_mean"]
                                    if "offpolicy_loss" in info_mean:
                                        wandb_log["train/offpolicy_loss"] = info_mean["offpolicy_loss"]
                                    if "offpolicy_active_ratio" in info_mean:
                                        wandb_log["train/offpolicy_active_ratio"] = info_mean["offpolicy_active_ratio"]
                                    if "offpolicy_ratio_mean" in info_mean:
                                        wandb_log["train/offpolicy_ratio_mean"] = info_mean["offpolicy_ratio_mean"]
                                    if "offpolicy_adv_mean" in info_mean:
                                        wandb_log["train/offpolicy_adv_mean"] = info_mean["offpolicy_adv_mean"]
                                    if "offpolicy_logprob_mean" in info_mean:
                                        wandb_log["train/offpolicy_logprob_mean"] = info_mean["offpolicy_logprob_mean"]
                                    wandb_log.update(adaptive_grad_clipper.state_dict())
                                    wandb.log(wandb_log, step=global_step)
                            info = defaultdict(list)

            if backward_counter % accum_steps_total != 0:
                grad_norm = adaptive_grad_clipper.adaptive_clip(trainable_parameters)
                optimizer.step()
                model.zero_grad(set_to_none=True)
                ema_model.update(model, global_step + 1)
                global_step += 1
                backward_counter = 0

                if len(info) > 0:
                    info_mean = {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}
                    if rank == 0:
                        tqdm.write(
                            f"  step {global_step} (tail) | loss: {info_mean.get('loss', 0):.6f} | "
                            f"policy_loss: {info_mean.get('policy_loss', 0):.6f} | "
                            f"gt_teacher_loss: {info_mean.get('gt_teacher_loss', 0):.6f} | "
                            f"offpolicy_loss: {info_mean.get('offpolicy_loss', 0):.6f} | "
                            f"approx_kl: {info_mean.get('approx_kl', 0):.6f} | "
                            f"clipfrac: {info_mean.get('clipfrac', 0):.4f} | "
                            f"grad_norm: {grad_norm.item():.4f}"
                        )
                        if wandb.run is not None:
                            wandb_log = {
                                "train/loss": info_mean.get("loss", 0),
                                "train/policy_loss": info_mean.get("policy_loss", 0),
                                "train/approx_kl": info_mean.get("approx_kl", 0),
                                "train/clipfrac": info_mean.get("clipfrac", 0),
                                "train/grad_norm": grad_norm.item(),
                                "train/lr": optimizer.param_groups[0]['lr'],
                            }
                            if "kl_loss" in info_mean:
                                wandb_log["train/kl_loss"] = info_mean["kl_loss"]
                            if "gt_teacher_loss" in info_mean:
                                wandb_log["train/gt_teacher_loss"] = info_mean["gt_teacher_loss"]
                            if "gt_teacher_active_ratio" in info_mean:
                                wandb_log["train/gt_teacher_active_ratio"] = info_mean["gt_teacher_active_ratio"]
                            if "gt_teacher_adv_mean" in info_mean:
                                wandb_log["train/gt_teacher_adv_mean"] = info_mean["gt_teacher_adv_mean"]
                            if "offpolicy_loss" in info_mean:
                                wandb_log["train/offpolicy_loss"] = info_mean["offpolicy_loss"]
                            if "offpolicy_active_ratio" in info_mean:
                                wandb_log["train/offpolicy_active_ratio"] = info_mean["offpolicy_active_ratio"]
                            if "offpolicy_ratio_mean" in info_mean:
                                wandb_log["train/offpolicy_ratio_mean"] = info_mean["offpolicy_ratio_mean"]
                            if "offpolicy_adv_mean" in info_mean:
                                wandb_log["train/offpolicy_adv_mean"] = info_mean["offpolicy_adv_mean"]
                            if "offpolicy_logprob_mean" in info_mean:
                                wandb_log["train/offpolicy_logprob_mean"] = info_mean["offpolicy_logprob_mean"]
                            wandb_log.update(adaptive_grad_clipper.state_dict())
                            wandb.log(wandb_log, step=global_step)
                    info = defaultdict(list)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier()

        if epoch > 0 and epoch % save_interval == 0:
            log_on_main_process(logger, f"Saving LoRA checkpoint at epoch {epoch} (global_step {global_step})...")
            if hasattr(model, 'save_pretrained'):
                save_lora_checkpoint(model, output_dir, global_step)
                ema_model.store(model)
                ema_model.ema_copy_to_model(model)
                save_lora_checkpoint(model, output_dir, global_step, suffix="-ema")
                ema_model.restore(model)
            adaptive_grad_clipper.save(output_dir=_lora_checkpoint_dir(output_dir, global_step))
            save_rl_training_state(output_dir, global_step, epoch + 1)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        last_completed_epoch = epoch

        if (test_dataloader is not None
                and global_step > 0
                and global_step % eval_freq == 0):
            log_on_main_process(logger, f"[Eval] Running eval video generation at global_step {global_step}...")
            model.eval()

            ema_model.store(model)
            ema_model.ema_copy_to_model(model)

            if reshard_after_forward is not None and not reshard_after_forward:
                model.set_reshard_after_forward(True, recurse=True)

            eval_videos_cpu = []
            eval_prompts = []

            for eval_batch in tqdm(test_dataloader, desc=f"[Eval] Generating videos", disable=(rank != 0)):
                eval_prompt_texts = eval_batch[PROMPT]
                eval_prompt_ids = eval_batch[PROMPT_IDS].to(device)
                eval_prompt_mask = eval_batch[PROMPT_MASK].to(device)

                if encoder_cpu_offload:
                    vae.model.to(device)
                    if not text_encoder_use_fsdp:
                        text_encoder.model.to(device)

                with torch.no_grad():
                    eval_text_embeddings = text_encoder(eval_prompt_ids, eval_prompt_mask)
                torch.cuda.synchronize()

                if encoder_cpu_offload:
                    vae.model.to("cpu")
                    if not text_encoder_use_fsdp:
                        text_encoder.model.to("cpu")
                    torch.cuda.empty_cache()

                eval_latent_shape = (len(eval_prompt_texts), latent_C, latent_T, latent_H, latent_W)

                with torch.no_grad():
                    eval_videos = osp_sample_deterministic(
                        model=model,
                        scheduler=scheduler,
                        vae=vae,
                        latent_shape=eval_latent_shape,
                        text_embeddings=eval_text_embeddings,
                        device=device,
                        weight_dtype=weight_dtype,
                        num_inference_steps=eval_num_steps,
                        guidance_scale=guidance_scale,
                        negative_text_embeddings=neg_text_embeddings.expand(len(eval_prompt_texts), -1, -1),
                        start_frame_latents=None,
                    )

                eval_videos_cpu.append(eval_videos.detach().cpu())
                eval_prompts.extend(eval_prompt_texts)

                del eval_videos, eval_text_embeddings, eval_prompt_ids, eval_prompt_mask
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if reshard_after_forward is not None and not reshard_after_forward:
                model.set_reshard_after_forward(reshard_after_forward, recurse=True)

            ema_model.restore(model)

            if rank == 0 and len(eval_videos_cpu) > 0:
                eval_all_videos = torch.cat(eval_videos_cpu, dim=0)
                with tempfile.TemporaryDirectory() as tmpdir:
                    num_eval_vis = len(eval_all_videos)
                    for idx in range(num_eval_vis):
                        video = eval_all_videos[idx].numpy().transpose(1, 2, 3, 0)
                        frames = [((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8) for frame in video]
                        imageio.mimsave(
                            os.path.join(tmpdir, f"eval_{idx}.mp4"),
                            frames, fps=16, codec="libx264", format='FFMPEG',
                        )

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "eval/videos": [
                                    wandb.Video(
                                        os.path.join(tmpdir, f"eval_{idx}.mp4"),
                                        caption=f"{eval_prompts[idx]:.100}",
                                        fps=16,
                                    )
                                    for idx in range(num_eval_vis)
                                ],
                            },
                            step=global_step,
                        )
                del eval_all_videos
            del eval_videos_cpu, eval_prompts
            torch.cuda.empty_cache()

            log_on_main_process(logger, f"[Eval] Eval video generation done at global_step {global_step}.")

    completed_epochs = max(start_epoch, last_completed_epoch + 1)
    log_on_main_process(logger, f"Saving final LoRA checkpoint at global_step {global_step}...")
    if hasattr(model, 'save_pretrained'):
        save_lora_checkpoint(model, output_dir, global_step)
        ema_model.store(model)
        ema_model.ema_copy_to_model(model)
        save_lora_checkpoint(model, output_dir, global_step, suffix="-ema")
        ema_model.restore(model)
    adaptive_grad_clipper.save(output_dir=_lora_checkpoint_dir(output_dir, global_step))
    save_rl_training_state(output_dir, global_step, completed_epochs)

    log_on_main_process(logger, f"""
    {'=' * 20}End RL Training (LoRA + FSDP2){'=' * 20}
    Total epochs: {completed_epochs}
    Total global steps: {global_step}
    Model saved to {output_dir}
    {'=' * 20}{'=' * len('End RL Training (LoRA + FSDP2)')}{'=' * 20}
    """)
    cleanup_distributed_env()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/gpu/osp_14b_RL_lora.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError(f"Config file {args.config} does not exist!")
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)
