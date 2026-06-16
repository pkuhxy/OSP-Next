import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _to_jsonable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _extract_reward(trace):
    reward = trace.get("reward", None)
    if reward is None:
        return None
    return _to_jsonable(reward)


def export_trace_metadata(trace_dir, output_path=None, max_workers=None):
    trace_paths = _discover_trace_files(trace_dir)
    if output_path is None:
        output_path = os.path.join(trace_dir, "trace_index.jsonl")
    if max_workers is None:
        max_workers = min(16, max(1, os.cpu_count() or 4))

    def _load_one(trace_path):
        trace = torch.load(trace_path, map_location="cpu")
        steps = trace.get("steps", [])
        return {
            "trace_path": trace_path,
            "prompt": trace.get("prompt", None),
            "seed": _to_jsonable(trace.get("seed", None)),
            "sample_index": _to_jsonable(trace.get("sample_index", None)),
            "x_T_shape": list(trace["x_T"].shape) if isinstance(trace.get("x_T"), torch.Tensor) else None,
            "num_steps": len(steps),
            "teacher_logprob_mode": trace.get("teacher_logprob_mode", None),
            "reward": _extract_reward(trace),
            "decoded_video_path": trace.get("decoded_video_path", None),
            "final_latents_shape": list(trace["final_latents"].shape) if isinstance(trace.get("final_latents"), torch.Tensor) else None,
            "sigmas_shape": list(trace["sigmas"].shape) if isinstance(trace.get("sigmas"), torch.Tensor) else None,
            "timesteps_shape": list(trace["timesteps"].shape) if isinstance(trace.get("timesteps"), torch.Tensor) else None,
        }

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_load_one, trace_path): trace_path for trace_path in trace_paths}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Exporting trace metadata"):
            entry = future.result()
            results[entry["trace_path"]] = entry

    with open(output_path, "w", encoding="utf-8") as f:
        for trace_path in trace_paths:
            f.write(json.dumps(results[trace_path], ensure_ascii=False) + "\n")

    print(f"Saved trace metadata to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trace metadata from saved .pt files.")
    parser.add_argument("--trace_dir", required=True, help="Directory containing trace_*.pt files.")
    parser.add_argument("--output_path", default=None, help="Output jsonl path. Defaults to trace_dir/trace_index.jsonl")
    parser.add_argument("--max_workers", type=int, default=None, help="Thread pool size for loading .pt files.")
    args = parser.parse_args()

    export_trace_metadata(
        trace_dir=args.trace_dir,
        output_path=args.output_path,
        max_workers=args.max_workers,
    )
