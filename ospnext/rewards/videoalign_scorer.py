import os
import tempfile
import shutil
import numpy as np
import torch
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional
from PIL import Image
from tqdm import tqdm

class VideoAlignScorer:
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_from_pretrained: str = None,
        load_from_pretrained_step: int = -1,
        reward_dim: str = "Overall",  # "VQ", "MQ", "TA", "Overall"
        use_norm: bool = True,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        max_pixels: Optional[int] = None,
        save_fps: int = 8,  # fps for writing temp video files
    ):
        self.device = device
        self.reward_dim = reward_dim
        self.use_norm = use_norm
        self.fps = fps
        self.num_frames = num_frames
        self.max_pixels = max_pixels
        self.save_fps = save_fps

        # Import from local module (no sys.path hack needed)
        from ospnext.rewards.videoalign_inference import VideoVLMRewardInference

        self.inferencer = VideoVLMRewardInference(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            device=device,
            dtype=dtype,
        )

        print(f"[VideoAlignScorer] Initialized with checkpoint: {load_from_pretrained}")
        print(f"[VideoAlignScorer] reward_dim={reward_dim}, use_norm={use_norm}")

    def _save_video_to_tempfile(self, frames: np.ndarray, tmp_dir: str, idx: int) -> str:
        from torchvision.io import write_video

        video_path = os.path.join(tmp_dir, f"tmp_video_{idx}.mp4")
        video_tensor = torch.from_numpy(frames).to(torch.uint8)
        write_video(video_path, video_tensor, self.save_fps, video_codec="h264")
        return video_path

    def _convert_to_video_frames(self, images) -> List[np.ndarray]:
        result = []
        if isinstance(images, np.ndarray):
            if images.ndim == 5:
                for i in range(images.shape[0]):
                    result.append(images[i])
            elif images.ndim == 4:
                for i in range(images.shape[0]):
                    result.append(images[i:i+1])
        elif isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy() if images.is_cuda else images.numpy()
            return self._convert_to_video_frames(images_np)
        elif isinstance(images, (list, tuple)):
            for img in images:
                if isinstance(img, np.ndarray):
                    if img.ndim == 4:
                        result.append(img)
                    elif img.ndim == 3:
                        result.append(img[np.newaxis])
                elif isinstance(img, Image.Image):
                    arr = np.array(img)
                    result.append(arr[np.newaxis])
                elif isinstance(img, torch.Tensor):
                    arr = img.cpu().numpy() if img.is_cuda else img.numpy()
                    if arr.ndim == 4:
                        result.append(arr)
                    elif arr.ndim == 3:
                        result.append(arr[np.newaxis])
        return result

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List, np.ndarray, torch.Tensor],
        prompts: List[str],
    ) -> List[float]:
        """Backward compatibility for raw frames input."""
        video_frames_list = self._convert_to_video_frames(images)
        assert len(video_frames_list) == len(prompts), \
            f"Number of videos ({len(video_frames_list)}) must match prompts ({len(prompts)})"

        tmp_dir = tempfile.mkdtemp(prefix="videoalign_")
        try:
            video_paths = []
            for i, frames in enumerate(video_frames_list):
                if frames.dtype != np.uint8:
                    frames = np.clip(frames * 255 if frames.max() <= 1.0 else frames, 0, 255).astype(np.uint8)
                path = self._save_video_to_tempfile(frames, tmp_dir, i)
                video_paths.append(path)

            return self.score_files(video_paths, prompts)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @torch.no_grad()
    def score_files(self, video_paths: List[str], prompts: List[str]) -> List[float]:
        """
        Directly compute VideoAlign reward scores for existing video files.
        """
        assert len(video_paths) == len(prompts), \
            f"Number of video files ({len(video_paths)}) must match prompts ({len(prompts)})"
            
        rewards = self.inferencer.reward(
            video_paths=video_paths,
            prompts=prompts,
            fps=self.fps,
            num_frames=self.num_frames,
            max_pixels=self.max_pixels,
            use_norm=self.use_norm,
        )

        scores = [r[self.reward_dim] for r in rewards]
        return scores


def process_video_folders(main_dir: str, prompt_file: str, scorer: VideoAlignScorer):
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt txt file not found: {prompt_file}")
        
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Loaded {len(prompts)} prompts from {prompt_file}.")

    for subdir_name in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir_name)

        if not os.path.isdir(subdir_path):
            continue
            
        video_paths = []
        batch_prompts = []
        video_names = []

        for i, prompt in enumerate(prompts):
            vid_name = f"video_{i}.mp4"
            vid_path = os.path.join(subdir_path, vid_name)
            
            if os.path.exists(vid_path):
                video_paths.append(vid_path)
                batch_prompts.append(prompt)
                video_names.append(vid_name)
                
        if not video_paths:
            print(f"No corresponding videos found in {subdir_path}, skipping.")
            continue
            
        print(f"\nProcessing {len(video_paths)} videos in {subdir_path}...")

        try:
            scores = scorer.score_files(video_paths, batch_prompts)

            results = {}
            for v_name, score in zip(video_names, scores):
                results[v_name] = float(score)
                
            avg_score = sum(scores) / len(scores)
            
            output_data = {
                "scores": results,
                "average_score": float(avg_score)
            }
            
            out_json = os.path.join(subdir_path, "videoalign_scores.json")
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
                
            print(f"Success! Saved results to {out_json} (Average Score: {avg_score:.4f})")
            
        except Exception as e:
            print(f"Error processing {subdir_path}: {e}")


def _discover_trace_files(trace_dir: str):
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


def _has_reward(trace):
    reward = trace.get("reward", None)
    if reward is None:
        return False
    if isinstance(reward, dict):
        return reward.get("avg", None) is not None or reward.get("videoalign", None) is not None
    return True


def _atomic_save_trace(trace, trace_path: str):
    tmp_path = f"{trace_path}.tmp"
    torch.save(trace, tmp_path)
    os.replace(tmp_path, trace_path)


def _trace_manifest_path(trace_dir: str) -> str:
    return os.path.join(trace_dir, "trace_index.jsonl")


def _write_trace_manifest(trace_dir: str, entries):
    manifest_path = _trace_manifest_path(trace_dir)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp_path, manifest_path)


def _score_trace_video_tensor(scorer: VideoAlignScorer, trace, prompt: str):
    decoded_video = trace.get("decoded_video", None)
    if decoded_video is None:
        raise ValueError(
            "Trace has no existing decoded_video_path and no decoded_video tensor. "
            "Regenerate traces with infer_osp.py so decoded_video is saved."
        )
    video = torch.as_tensor(decoded_video).cpu()
    score = scorer(video.unsqueeze(0).numpy(), [prompt])[0]
    return float(score)


def score_trace_files(
    trace_dir: str,
    scorer: VideoAlignScorer,
    batch_size: int = 1,
    overwrite: bool = False,
    prefer_video_path: bool = True,
):
    """Score denoising trace .pt files and write VideoAlign reward into each trace.

    infer/infer_osp.py saves one sample per .pt file. This function writes:
        trace["reward"] = {"videoalign": score, "avg": score}

    Existing rewards are preserved unless overwrite=True.
    """
    trace_paths = _discover_trace_files(trace_dir)
    to_score = []
    skipped = 0
    manifest_entries = []
    for trace_path in tqdm(trace_paths, desc="Scanning traces"):
        trace = torch.load(trace_path, map_location="cpu")
        prompt = trace.get("prompt", None)
        if not prompt:
            raise ValueError(f"Trace {trace_path} has no prompt.")
        decoded_video_path = trace.get("decoded_video_path", None)
        can_use_video_path = (
            prefer_video_path
            and decoded_video_path is not None
            and os.path.exists(decoded_video_path)
        )
        manifest_entries.append({
            "trace_path": trace_path,
            "sample_index": trace.get("sample_index", None),
            "prompt": prompt,
            "reward": trace.get("reward", None),
            "decoded_video_path": decoded_video_path,
            "teacher_logprob_mode": trace.get("teacher_logprob_mode", None),
        })
        if _has_reward(trace) and not overwrite:
            skipped += 1
            continue
        to_score.append({
            "trace_path": trace_path,
            "prompt": prompt,
            "decoded_video_path": decoded_video_path if can_use_video_path else None,
        })

    print(
        f"Found {len(trace_paths)} trace files. "
        f"Need scoring: {len(to_score)}. Skipped existing rewards: {skipped}."
    )
    if not to_score:
        _write_trace_manifest(trace_dir, manifest_entries)
        print(f"Saved trace manifest to {_trace_manifest_path(trace_dir)}")
        return

    for start in tqdm(range(0, len(to_score), batch_size), desc="Scoring trace rewards"):
        batch = to_score[start:start + batch_size]
        path_items = [item for item in batch if item["decoded_video_path"] is not None]
        tensor_items = [item for item in batch if item["decoded_video_path"] is None]

        scores_by_path = {}
        if path_items:
            scores = scorer.score_files(
                [item["decoded_video_path"] for item in path_items],
                [item["prompt"] for item in path_items],
            )
            for item, score in zip(path_items, scores):
                scores_by_path[item["trace_path"]] = float(score)

        for item in tensor_items:
            trace = torch.load(item["trace_path"], map_location="cpu")
            scores_by_path[item["trace_path"]] = _score_trace_video_tensor(
                scorer=scorer,
                trace=trace,
                prompt=item["prompt"],
            )

        for item in batch:
            trace_path = item["trace_path"]
            trace = torch.load(trace_path, map_location="cpu")
            score = scores_by_path[trace_path]
            trace["reward"] = {
                "videoalign": float(score),
                "avg": float(score),
            }
            _atomic_save_trace(trace, trace_path)
            for entry in manifest_entries:
                if entry["trace_path"] == trace_path:
                    entry["reward"] = trace["reward"]
                    break

    print(f"Done. Wrote VideoAlign rewards into {len(to_score)} trace files.")
    _write_trace_manifest(trace_dir, manifest_entries)
    print(f"Saved trace manifest to {_trace_manifest_path(trace_dir)}")


def export_trace_index_from_rewards(trace_dir: str):
    """Export a lightweight trace_index.jsonl from existing trace rewards.

    This does not run VideoAlign. It only reads reward fields already stored in
    the trace .pt files.
    """
    trace_paths = _discover_trace_files(trace_dir)
    manifest_entries = []
    max_workers = min(16, max(1, (os.cpu_count() or 4)))

    def _load_one(trace_path):
        trace = torch.load(trace_path, map_location="cpu")
        prompt = trace.get("prompt", None)
        if not prompt:
            raise ValueError(f"Trace {trace_path} has no prompt.")
        reward = trace.get("reward", None)
        return {
            "trace_path": trace_path,
            "sample_index": trace.get("sample_index", None),
            "prompt": prompt,
            "reward": None if reward is None else _trace_reward_to_scalar(reward),
            "decoded_video_path": trace.get("decoded_video_path", None),
            "teacher_logprob_mode": trace.get("teacher_logprob_mode", None),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_load_one, trace_path): trace_path
            for trace_path in trace_paths
        }
        results = {}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Exporting trace index"):
            entry = future.result()
            results[entry["trace_path"]] = entry

    manifest_entries = [results[trace_path] for trace_path in trace_paths]
    _write_trace_manifest(trace_dir, manifest_entries)
    print(f"Saved trace manifest to {_trace_manifest_path(trace_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and calculate VideoAlign scores.")
    parser.add_argument("--main_dir", type=str, default="/home/ma-user/work/xianyi/osp_next/ospnext/samples/osp_next_mixgrpo/moviegen/lr104_bf16_2", help="Path to the main folder containing subfolders of videos.")
    parser.add_argument("--prompt_file", type=str, default="/home/ma-user/work/xianyi/osp_next/ospnext/assets/t2v/eval_Moviegen.txt", help="Path to the txt file containing prompts.")
    parser.add_argument("--trace_dir", type=str, default=None, help="Path to infer_osp.py denoising trace .pt files. If set, rewards are written back into each trace.")
    parser.add_argument("--ckpt_path", type=str, default="/home/ma-user/work/xianyi/ckpts/KlingTeam/VideoReward", help="Path to VideoAlign checkpoint.")
    parser.add_argument("--device", type=str, default="npu", help="Device to run on (e.g., cuda, npu).")
    parser.add_argument("--reward_dim", type=str, default="Overall", choices=["VQ", "MQ", "TA", "Overall"], help="Reward dimension.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for VideoAlign scoring.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rewards in trace .pt files.")
    parser.add_argument("--no_prefer_video_path", action="store_true", help="Ignore decoded_video_path and score decoded_video tensors from trace files.")
    parser.add_argument("--export_trace_index_only", action="store_true", help="Only export trace_index.jsonl from existing trace rewards without running VideoAlign.")
    
    args = parser.parse_args()

    scorer = None
    if not args.export_trace_index_only:
        scorer = VideoAlignScorer(
            device=args.device,
            load_from_pretrained=args.ckpt_path,
            reward_dim=args.reward_dim,
        )

    if args.trace_dir is not None and args.export_trace_index_only:
        export_trace_index_from_rewards(args.trace_dir)
    elif args.trace_dir is not None:
        score_trace_files(
            trace_dir=args.trace_dir,
            scorer=scorer,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            prefer_video_path=not args.no_prefer_video_path,
        )
    else:
        if scorer is None:
            scorer = VideoAlignScorer(
                device=args.device,
                load_from_pretrained=args.ckpt_path,
                reward_dim=args.reward_dim,
            )
        process_video_folders(
            main_dir=args.main_dir,
            prompt_file=args.prompt_file,
            scorer=scorer
        )
