from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


IMAGE_MEDIA_DEFAULTS: Dict[str, Any] = {
    "min_pixels": 4096,
    "max_pixels": 16777216,
    "multi_image_max_pixels": 201326592,
    "patch_size": 16,
    "temporal_patch_size": 1,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}

VIDEO_MEDIA_DEFAULTS: Dict[str, Any] = {
    "min_pixels": 4096,
    "max_pixels": 16777216,
    "video_max_pixels": 201326592,
    "patch_size": 16,
    "temporal_patch_size": 1,
    "merge_size": 2,
    "video_fps": 1.0,
    "min_frames": 1,
    "max_frames": 256,
    "num_extract_threads": 4,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}

GENERATE_DEFAULTS: Dict[str, Any] = {
    "max_new_tokens": 256,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "do_sample": False,
    "vision_chunked_length": 64,
}

OPTIONAL_QUERY_KEYS = (
    "thinking_mode",
    "system_prompt_type",
    "system_prompt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOSS-VL inference with README-aligned entry points.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the MOSS-VL checkpoint directory.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("image", "video", "batch"),
        help="Inference mode: image, video, or batch.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save output JSON. Defaults to <input>_results.json.",
    )
    return parser.parse_args()


def load_model(checkpoint: str):
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        frame_extract_num_threads=1,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return model, processor


def load_queries(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {input_path}, but got {type(data).__name__}.")
    return data


def resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def resolve_video_entry(base_dir: Path, value: Any) -> Any:
    if isinstance(value, str):
        return resolve_path(base_dir, value)
    if isinstance(value, dict):
        resolved = dict(value)
        if "video_path" not in resolved:
            raise ValueError(f"Video dict is missing `video_path`: {value}")
        resolved["video_path"] = resolve_path(base_dir, resolved["video_path"])
        return resolved
    raise ValueError(f"Unsupported video entry type: {type(value).__name__}")


def resolve_query_media_paths(query: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    resolved = dict(query)
    resolved["images"] = [resolve_path(base_dir, image) for image in query.get("images", [])]
    resolved["videos"] = [resolve_video_entry(base_dir, video) for video in query.get("videos", [])]
    resolved["media_kwargs"] = dict(query.get("media_kwargs") or {})
    resolved["generate_kwargs"] = dict(query.get("generate_kwargs") or {})
    return resolved


def ensure_mode_query_shape(mode: str, query: Dict[str, Any], index: int) -> None:
    images = query.get("images") or []
    videos = query.get("videos") or []
    if mode == "image":
        if len(images) != 1 or videos:
            raise ValueError(
                f"Image mode expects exactly one image and zero videos for query #{index}, "
                f"but got {len(images)} image(s) and {len(videos)} video(s)."
            )
    elif mode == "video":
        if len(videos) != 1 or images:
            raise ValueError(
                f"Video mode expects exactly one video and zero images for query #{index}, "
                f"but got {len(images)} image(s) and {len(videos)} video(s)."
            )


def merge_defaults(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(overrides or {})
    return merged


def validate_batch_shared_fields(queries: List[Dict[str, Any]]) -> None:
    if not queries:
        return

    def collect_mismatched_keys(field_name: str) -> List[str]:
        values = [dict(query.get(field_name) or {}) for query in queries]
        all_keys = set()
        for value in values:
            all_keys.update(value.keys())

        mismatched: List[str] = []
        for key in sorted(all_keys):
            unique_values = {repr(value.get(key)) for value in values}
            if len(unique_values) > 1:
                mismatched.append(key)
        return mismatched

    mismatched_media = collect_mismatched_keys("media_kwargs")
    mismatched_generate = collect_mismatched_keys("generate_kwargs")
    if mismatched_media or mismatched_generate:
        parts = []
        if mismatched_media:
            parts.append(f"media_kwargs: {', '.join(mismatched_media)}")
        if mismatched_generate:
            parts.append(f"generate_kwargs: {', '.join(mismatched_generate)}")
        raise ValueError(
            "Batch mode requires all queries to share the same configuration. "
            + "; ".join(parts)
        )


def build_single_result(
    index: int,
    original_query: Dict[str, Any],
    response_text: str,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    return {
        "index": index,
        "prompt": original_query.get("prompt", ""),
        "images": original_query.get("images", []),
        "videos": original_query.get("videos", []),
        "media_kwargs": original_query.get("media_kwargs", {}),
        "generate_kwargs": original_query.get("generate_kwargs", {}),
        "text": response_text,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def apply_optional_query_args(target_kwargs: Dict[str, Any], query: Dict[str, Any]) -> None:
    for key in OPTIONAL_QUERY_KEYS:
        if query.get(key) is not None:
            target_kwargs[key] = query[key]


def run_image_queries(model, processor, original_queries: List[Dict[str, Any]], input_dir: Path) -> List[Dict[str, Any]]:
    results = []
    for index, original_query in enumerate(original_queries):
        resolved_query = resolve_query_media_paths(original_query, input_dir)
        ensure_mode_query_shape("image", resolved_query, index)

        media_kwargs = merge_defaults(IMAGE_MEDIA_DEFAULTS, resolved_query["media_kwargs"])
        generate_kwargs = merge_defaults(GENERATE_DEFAULTS, resolved_query["generate_kwargs"])
        call_kwargs: Dict[str, Any] = {
            "prompt": resolved_query.get("prompt", ""),
            "image": resolved_query["images"][0],
            "shortest_edge": media_kwargs["min_pixels"],
            "longest_edge": media_kwargs["max_pixels"],
            "multi_image_max_pixels": media_kwargs["multi_image_max_pixels"],
            "patch_size": media_kwargs["patch_size"],
            "temporal_patch_size": media_kwargs["temporal_patch_size"],
            "merge_size": media_kwargs["merge_size"],
            "image_mean": media_kwargs["image_mean"],
            "image_std": media_kwargs["image_std"],
            "max_new_tokens": generate_kwargs["max_new_tokens"],
            "temperature": generate_kwargs["temperature"],
            "top_k": generate_kwargs["top_k"],
            "top_p": generate_kwargs["top_p"],
            "repetition_penalty": generate_kwargs["repetition_penalty"],
            "do_sample": generate_kwargs["do_sample"],
            "vision_chunked_length": generate_kwargs["vision_chunked_length"],
        }
        apply_optional_query_args(call_kwargs, resolved_query)

        start_time = time.time()
        text = model.offline_image_generate(processor, **call_kwargs)
        elapsed_seconds = time.time() - start_time
        results.append(build_single_result(index, original_query, text, elapsed_seconds))
    return results


def run_video_queries(model, processor, original_queries: List[Dict[str, Any]], input_dir: Path) -> List[Dict[str, Any]]:
    results = []
    for index, original_query in enumerate(original_queries):
        resolved_query = resolve_query_media_paths(original_query, input_dir)
        ensure_mode_query_shape("video", resolved_query, index)

        media_kwargs = merge_defaults(VIDEO_MEDIA_DEFAULTS, resolved_query["media_kwargs"])
        generate_kwargs = merge_defaults(GENERATE_DEFAULTS, resolved_query["generate_kwargs"])
        call_kwargs: Dict[str, Any] = {
            "prompt": resolved_query.get("prompt", ""),
            "video": resolved_query["videos"][0],
            "shortest_edge": media_kwargs["min_pixels"],
            "longest_edge": media_kwargs["max_pixels"],
            "video_max_pixels": media_kwargs["video_max_pixels"],
            "patch_size": media_kwargs["patch_size"],
            "temporal_patch_size": media_kwargs["temporal_patch_size"],
            "merge_size": media_kwargs["merge_size"],
            "video_fps": media_kwargs["video_fps"],
            "min_frames": media_kwargs["min_frames"],
            "max_frames": media_kwargs["max_frames"],
            "num_extract_threads": media_kwargs["num_extract_threads"],
            "image_mean": media_kwargs["image_mean"],
            "image_std": media_kwargs["image_std"],
            "max_new_tokens": generate_kwargs["max_new_tokens"],
            "temperature": generate_kwargs["temperature"],
            "top_k": generate_kwargs["top_k"],
            "top_p": generate_kwargs["top_p"],
            "repetition_penalty": generate_kwargs["repetition_penalty"],
            "do_sample": generate_kwargs["do_sample"],
            "vision_chunked_length": generate_kwargs["vision_chunked_length"],
        }
        apply_optional_query_args(call_kwargs, resolved_query)

        start_time = time.time()
        text = model.offline_video_generate(processor, **call_kwargs)
        elapsed_seconds = time.time() - start_time
        results.append(build_single_result(index, original_query, text, elapsed_seconds))
    return results


def run_batch_queries(model, processor, original_queries: List[Dict[str, Any]], input_dir: Path) -> Dict[str, Any]:
    resolved_queries = [resolve_query_media_paths(query, input_dir) for query in original_queries]
    validate_batch_shared_fields(resolved_queries)

    start_time = time.time()
    output = model.offline_batch_generate(processor, resolved_queries)
    elapsed_seconds = time.time() - start_time

    results = []
    raw_results = output.get("results", [])
    for item in raw_results:
        index = item["index"]
        original_query = original_queries[index]
        results.append(
            {
                "index": index,
                "prompt": original_query.get("prompt", ""),
                "images": original_query.get("images", []),
                "videos": original_query.get("videos", []),
                "media_kwargs": original_query.get("media_kwargs", {}),
                "generate_kwargs": original_query.get("generate_kwargs", {}),
                "text": item.get("text", ""),
                "input_text": item.get("input_text", ""),
            }
        )

    return {
        "results": results,
        "session_states": output.get("session_states", []),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_results.json")


def save_json(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else default_output_path(input_path)
    )

    queries = load_queries(input_path)
    model, processor = load_model(checkpoint)

    if args.mode == "image":
        results = run_image_queries(model, processor, queries, input_path.parent)
        payload: Dict[str, Any] = {
            "mode": "image",
            "checkpoint": checkpoint,
            "input": str(input_path),
            "results": results,
        }
    elif args.mode == "video":
        results = run_video_queries(model, processor, queries, input_path.parent)
        payload = {
            "mode": "video",
            "checkpoint": checkpoint,
            "input": str(input_path),
            "results": results,
        }
    else:
        batch_output = run_batch_queries(model, processor, queries, input_path.parent)
        payload = {
            "mode": "batch",
            "checkpoint": checkpoint,
            "input": str(input_path),
            **batch_output,
        }

    save_json(output_path, payload)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
