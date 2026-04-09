from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Iterable, List

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

LEGACY_MODE_ALIASES = ("offline", "image", "video", "batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOSS-VL inference via the generic offline_generate API.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint directory, for example /path/to/dummy-checkpoint.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=LEGACY_MODE_ALIASES,
        help="Legacy label kept for compatibility. All modes now route through offline_generate.",
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
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout per query while waiting for offline_generate to finish.",
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


def resolve_message_content_media_paths(content: Any, base_dir: Path) -> Any:
    if not isinstance(content, list):
        return content

    resolved_content = []
    for item in content:
        if not isinstance(item, dict):
            resolved_content.append(item)
            continue

        item_copy = dict(item)
        if item_copy.get("type") == "image" or "image" in item_copy or "image_url" in item_copy:
            if item_copy.get("image") is not None:
                item_copy["image"] = resolve_path(base_dir, item_copy["image"])
            if item_copy.get("image_url") is not None:
                item_copy["image_url"] = resolve_path(base_dir, item_copy["image_url"])
        elif item_copy.get("type") == "video" or "video" in item_copy:
            if item_copy.get("video") is not None:
                item_copy["video"] = resolve_video_entry(base_dir, item_copy["video"])
        resolved_content.append(item_copy)
    return resolved_content


def resolve_query_media_paths(query: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    resolved = dict(query)
    resolved["images"] = [resolve_path(base_dir, image) for image in query.get("images", [])]
    resolved["videos"] = [resolve_video_entry(base_dir, video) for video in query.get("videos", [])]
    resolved["media_kwargs"] = dict(query.get("media_kwargs") or {})
    resolved["generate_kwargs"] = dict(query.get("generate_kwargs") or {})

    if query.get("messages") is not None:
        resolved_messages = []
        for message in query.get("messages") or []:
            if not isinstance(message, dict):
                continue
            message_copy = dict(message)
            message_copy["content"] = resolve_message_content_media_paths(message.get("content", ""), base_dir)
            resolved_messages.append(message_copy)
        resolved["messages"] = resolved_messages

    return resolved


def iter_message_media_items(messages: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict):
                yield item


def query_has_images(query: Dict[str, Any]) -> bool:
    if query.get("images"):
        return True
    return any(
        item.get("type") == "image" or "image" in item or "image_url" in item
        for item in iter_message_media_items(query.get("messages") or [])
    )


def query_has_videos(query: Dict[str, Any]) -> bool:
    if query.get("videos"):
        return True
    return any(
        item.get("type") == "video" or "video" in item
        for item in iter_message_media_items(query.get("messages") or [])
    )


def merge_defaults(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(overrides or {})
    return merged


def normalize_query_for_offline_generate(query: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    resolved = resolve_query_media_paths(query, base_dir)

    default_media_kwargs: Dict[str, Any] = {}
    if query_has_images(resolved):
        default_media_kwargs.update(IMAGE_MEDIA_DEFAULTS)
    if query_has_videos(resolved):
        default_media_kwargs.update(VIDEO_MEDIA_DEFAULTS)

    normalized = dict(resolved)
    normalized["media_kwargs"] = merge_defaults(default_media_kwargs, resolved["media_kwargs"])
    normalized["generate_kwargs"] = merge_defaults(GENERATE_DEFAULTS, resolved["generate_kwargs"])
    return normalized


def collect_offline_generate_output(
    output_queue: "queue.Queue[str]",
    timeout_seconds: float,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    chunks: List[str] = []
    events: List[str] = []
    error_text = None

    while time.time() < deadline:
        remaining = max(0.1, min(1.0, deadline - time.time()))
        try:
            item = output_queue.get(timeout=remaining)
        except queue.Empty:
            continue

        events.append(item)
        if item == "<|round_start|>":
            continue
        if item == "<|round_end|>":
            return {
                "text": "".join(chunks),
                "events": events,
                "error": error_text,
            }
        if item.startswith("[ERROR] "):
            error_text = item
            continue
        chunks.append(item)

    raise TimeoutError("Timed out waiting for offline_generate to finish the current round.")


def run_offline_generate_query(
    model,
    processor,
    query: Dict[str, Any],
    timeout_seconds: float,
) -> Dict[str, Any]:
    input_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    output_queue: "queue.Queue[str]" = queue.Queue()
    worker = Thread(
        target=model.offline_generate,
        args=(processor, input_queue, output_queue),
        kwargs={"vision_chunked_length": query["generate_kwargs"].get("vision_chunked_length", 64)},
        daemon=True,
    )
    worker.start()

    try:
        input_queue.put(dict(query))
        result = collect_offline_generate_output(output_queue, timeout_seconds)
    finally:
        input_queue.put({"stop_offline_generate": True})
        worker.join(timeout=30.0)

    if result["error"] is not None:
        raise RuntimeError(result["error"])
    return result


def build_single_result(
    index: int,
    original_query: Dict[str, Any],
    response_text: str,
    elapsed_seconds: float,
    events: List[str],
) -> Dict[str, Any]:
    return {
        "index": index,
        "prompt": original_query.get("prompt", ""),
        "messages": original_query.get("messages"),
        "images": original_query.get("images", []),
        "videos": original_query.get("videos", []),
        "media_kwargs": original_query.get("media_kwargs", {}),
        "generate_kwargs": original_query.get("generate_kwargs", {}),
        "text": response_text,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "event_count": len(events),
    }


def run_queries_via_offline_generate(
    model,
    processor,
    original_queries: List[Dict[str, Any]],
    input_dir: Path,
    timeout_seconds: float,
) -> Dict[str, Any]:
    results = []

    overall_start = time.time()
    for index, original_query in enumerate(original_queries):
        resolved_query = normalize_query_for_offline_generate(original_query, input_dir)

        start_time = time.time()
        run_result = run_offline_generate_query(
            model,
            processor,
            resolved_query,
            timeout_seconds=timeout_seconds,
        )
        elapsed_seconds = time.time() - start_time
        results.append(
            build_single_result(
                index=index,
                original_query=original_query,
                response_text=run_result["text"],
                elapsed_seconds=elapsed_seconds,
                events=run_result["events"],
            )
        )

    return {
        "results": results,
        "elapsed_seconds": round(time.time() - overall_start, 3),
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
    output = run_queries_via_offline_generate(
        model,
        processor,
        queries,
        input_path.parent,
        timeout_seconds=args.timeout_seconds,
    )

    payload: Dict[str, Any] = {
        "mode": "offline_generate",
        "mode_alias": args.mode,
        "checkpoint": checkpoint,
        "input": str(input_path),
        **output,
    }

    save_json(output_path, payload)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
