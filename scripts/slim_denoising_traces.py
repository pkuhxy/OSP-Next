import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm


def _discover_trace_files(trace_dir):
    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")
    trace_paths = [
        os.path.join(trace_dir, name)
        for name in sorted(os.listdir(trace_dir))
        if name.endswith((".pt", ".pth"))
    ]
    if not trace_paths:
        raise FileNotFoundError(f"No .pt/.pth trace files found in {trace_dir}")
    return trace_paths


def _load_trace_file(trace_path):
    try:
        return torch.load(trace_path, map_location="cpu", mmap=True)
    except (TypeError, ValueError, RuntimeError):
        return torch.load(trace_path, map_location="cpu")


def _trace_reward_to_scalar(reward):
    if reward is None:
        return None
    if isinstance(reward, dict):
        for key in ("avg", "mean", "score", "reward", "total", "videoalign"):
            if key in reward:
                return _trace_reward_to_scalar(reward[key])
        values = [_trace_reward_to_scalar(value) for value in reward.values()]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None
    if isinstance(reward, torch.Tensor):
        return float(reward.detach().cpu().float().mean().item())
    if isinstance(reward, np.ndarray):
        return float(np.asarray(reward, dtype=np.float32).mean())
    if isinstance(reward, (list, tuple)):
        return float(np.asarray(reward, dtype=np.float32).mean())
    return float(reward)


def _to_jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _convert_tensor(value, latent_dtype):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if latent_dtype == "fp32":
        return tensor.float()
    if latent_dtype == "fp16":
        return tensor.half()
    if latent_dtype == "bf16":
        return tensor.bfloat16()
    return tensor


def _slim_one(trace_path, output_dir, latent_dtype, overwrite):
    trace = _load_trace_file(trace_path)
    prompt = trace.get("prompt", None)
    if not prompt:
        raise ValueError(f"Trace {trace_path} has no prompt.")
    if "sigmas" not in trace:
        raise ValueError(f"Trace {trace_path} has no sigmas.")
    if "steps" not in trace:
        raise ValueError(f"Trace {trace_path} has no steps.")

    output_path = os.path.join(output_dir, os.path.basename(trace_path))
    if os.path.exists(output_path) and not overwrite:
        reward = trace.get("reward", None)
        return {
            "trace_path": os.path.abspath(output_path),
            "sample_index": _to_jsonable(trace.get("sample_index", None)),
            "prompt": prompt,
            "reward": None if reward is None else _trace_reward_to_scalar(reward),
            "decoded_video_path": trace.get("decoded_video_path", None),
            "teacher_logprob_mode": trace.get("teacher_logprob_mode", None),
        }

    slim_steps = []
    for step in trace["steps"]:
        teacher_logprob = step.get("teacher_logprob", None)
        if teacher_logprob is not None:
            teacher_logprob = torch.as_tensor(teacher_logprob, dtype=torch.float32).mean()
        slim_steps.append({
            "step_index": int(step["step_index"]),
            "sigma": _to_jsonable(step.get("sigma", None)),
            "sigma_next": _to_jsonable(step.get("sigma_next", None)),
            "timestep": _to_jsonable(step.get("timestep", None)),
            "x_t": _convert_tensor(step["x_t"], latent_dtype),
            "x_t_minus_1": _convert_tensor(step["x_t_minus_1"], latent_dtype),
            "teacher_logprob": teacher_logprob,
        })

    slim_trace = {
        "prompt": prompt,
        "seed": _to_jsonable(trace.get("seed", None)),
        "sample_index": _to_jsonable(trace.get("sample_index", None)),
        "sigmas": torch.as_tensor(trace["sigmas"], dtype=torch.float32).detach().cpu().contiguous(),
        "timesteps": torch.as_tensor(trace.get("timesteps", []), dtype=torch.float32).detach().cpu().contiguous(),
        "steps": slim_steps,
        "teacher_logprob_mode": trace.get("teacher_logprob_mode", None),
        "reward": trace.get("reward", None),
        "decoded_video_path": trace.get("decoded_video_path", None),
    }
    torch.save(slim_trace, output_path)

    reward = slim_trace.get("reward", None)
    return {
        "trace_path": os.path.abspath(output_path),
        "sample_index": slim_trace.get("sample_index", None),
        "prompt": prompt,
        "reward": None if reward is None else _trace_reward_to_scalar(reward),
        "decoded_video_path": slim_trace.get("decoded_video_path", None),
        "teacher_logprob_mode": slim_trace.get("teacher_logprob_mode", None),
    }


def slim_denoising_traces(trace_dir, output_dir, output_index=None, max_workers=4, latent_dtype="original", overwrite=False):
    trace_paths = _discover_trace_files(trace_dir)
    os.makedirs(output_dir, exist_ok=True)
    if output_index is None:
        output_index = os.path.join(output_dir, "trace_index.jsonl")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_slim_one, trace_path, output_dir, latent_dtype, overwrite): trace_path
            for trace_path in trace_paths
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Slimming traces"):
            trace_path = future_map[future]
            results[trace_path] = future.result()

    with open(output_index, "w", encoding="utf-8") as f:
        for trace_path in trace_paths:
            f.write(json.dumps(results[trace_path], ensure_ascii=False) + "\n")

    print(f"Saved slim traces to {output_dir}")
    print(f"Saved slim trace index to {output_index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create smaller off-policy denoising trace files for stage3 training.")
    parser.add_argument("--trace_dir", required=True, help="Directory containing full trace_*.pt files.")
    parser.add_argument("--output_dir", required=True, help="Directory for slim trace_*.pt files.")
    parser.add_argument("--output_index", default=None, help="Defaults to output_dir/trace_index.jsonl.")
    parser.add_argument("--max_workers", type=int, default=4, help="Thread pool size. Keep this modest on shared storage.")
    parser.add_argument("--latent_dtype", choices=("original", "fp32", "fp16", "bf16"), default="original")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing slim trace files.")
    args = parser.parse_args()

    slim_denoising_traces(
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
        output_index=args.output_index,
        max_workers=args.max_workers,
        latent_dtype=args.latent_dtype,
        overwrite=args.overwrite,
    )
