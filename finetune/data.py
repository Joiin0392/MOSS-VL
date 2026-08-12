"""Dataset and data collator for MOSS-VL supervised fine-tuning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASSISTANT_HEADER = "<|im_start|>assistant\n"
END_TOKEN = "<|im_end|>"
IMAGE_PLACEHOLDER = "<|image|>"
VIDEO_PLACEHOLDER = "<|video|>"


def find_assistant_spans(text: str) -> List[List[int]]:
    """Return character-level ``[start, end)`` spans covering each assistant
    response **including** the trailing ``<|im_end|>`` token so the model
    learns to produce the stop signal."""
    spans: List[List[int]] = []
    pos = 0
    while True:
        idx = text.find(ASSISTANT_HEADER, pos)
        if idx == -1:
            break
        content_start = idx + len(ASSISTANT_HEADER)
        end_idx = text.find(END_TOKEN, content_start)
        if end_idx == -1:
            spans.append([content_start, len(text)])
            break
        span_end = end_idx + len(END_TOKEN)
        spans.append([content_start, span_end])
        pos = span_end
    return spans


def _resolve_path(base_dir: str, value: Any) -> Any:
    """Resolve a media path (string) or video dict relative to *base_dir*."""
    if isinstance(value, str):
        p = Path(value)
        if not p.is_absolute():
            p = Path(base_dir) / p
        return str(p.resolve())
    if isinstance(value, dict):
        resolved = dict(value)
        if "video_path" in resolved:
            resolved["video_path"] = _resolve_path(base_dir, resolved["video_path"])
        return resolved
    return value


def _count_video_units(video_entry: Any) -> int:
    """Return how many `<|video|>` placeholders a video entry expands to.

    Plain video paths consume one placeholder. Segment dictionaries expand to
    one placeholder per segment because the downstream video processor flattens
    segments into independent media items.
    """
    if isinstance(video_entry, dict) and "segments" in video_entry:
        segments = video_entry.get("segments") or []
        return len(segments)
    return 1


def _count_video_placeholders_in_content(content: Any) -> int:
    if isinstance(content, str):
        return content.count(VIDEO_PLACEHOLDER)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += item.count(VIDEO_PLACEHOLDER)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                total += item["text"].count(VIDEO_PLACEHOLDER)
        return total
    return 0


def _expand_video_placeholders_in_text(
    text: str,
    video_units: List[int],
    video_index: int,
) -> tuple[str, int]:
    if VIDEO_PLACEHOLDER not in text:
        return text, video_index

    pieces = text.split(VIDEO_PLACEHOLDER)
    rebuilt = [pieces[0]]
    next_index = video_index
    for suffix in pieces[1:]:
        if next_index < len(video_units):
            rebuilt.append(VIDEO_PLACEHOLDER * video_units[next_index])
            next_index += 1
        else:
            rebuilt.append(VIDEO_PLACEHOLDER)
        rebuilt.append(suffix)
    return "".join(rebuilt), next_index


def _expand_video_placeholders_in_content(
    content: Any,
    video_units: List[int],
    video_index: int,
) -> tuple[Any, int]:
    if isinstance(content, str):
        return _expand_video_placeholders_in_text(content, video_units, video_index)
    if isinstance(content, list):
        expanded_content = []
        next_index = video_index
        for item in content:
            if isinstance(item, str):
                new_item, next_index = _expand_video_placeholders_in_text(
                    item, video_units, next_index
                )
                expanded_content.append(new_item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                new_item = dict(item)
                new_item["text"], next_index = _expand_video_placeholders_in_text(
                    item["text"], video_units, next_index
                )
                expanded_content.append(new_item)
            else:
                expanded_content.append(item)
        return expanded_content, next_index
    return content, video_index


def _expand_segment_video_placeholders(
    messages: List[Dict[str, Any]],
    videos: List[Any],
) -> List[Dict[str, Any]]:
    """Expand one `<|video|>` placeholder per segmented entry when needed.

    Conversation-format data may naturally contain one placeholder for each
    top-level video entry. When an entry contains `segments`, the downstream
    processor expands it into multiple media items, so we mirror that expansion
    in text only when the conversation still uses the pre-expansion count.
    """
    video_units = [_count_video_units(video) for video in videos]
    if not video_units:
        return list(messages)

    raw_video_count = len(video_units)
    expanded_video_count = sum(video_units)
    if expanded_video_count == raw_video_count:
        return list(messages)

    explicit_placeholder_count = sum(
        _count_video_placeholders_in_content(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )
    if explicit_placeholder_count != raw_video_count:
        return list(messages)

    expanded_messages: List[Dict[str, Any]] = []
    next_index = 0
    for message in messages:
        if not isinstance(message, dict):
            expanded_messages.append(message)
            continue
        message_copy = dict(message)
        message_copy["content"], next_index = _expand_video_placeholders_in_content(
            message_copy.get("content", ""),
            video_units,
            next_index,
        )
        expanded_messages.append(message_copy)
    return expanded_messages


def _build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a training sample to the chat-message list expected by the
    tokenizer's ``apply_chat_template``.

    Supports two formats:

    **Format 1 – prompt / response** (compatible with inference queries)::

        {
            "prompt": "Describe this image.",
            "response": "A landscape with …",
            "images": ["img.jpg"],
            "videos": []
        }

    Media placeholders (``<|image|>``, ``<|video|>``) are prepended
    automatically.

    **Format 2 – conversations** (multi-turn, explicit placeholders)::

        {
            "conversations": [
                {"role": "user", "content": "<|image|>\\nDescribe."},
                {"role": "assistant", "content": "A landscape …"}
            ],
            "images": ["img.jpg"],
            "videos": []
        }
    """
    if "conversations" in sample:
        messages = list(sample["conversations"])
        videos = sample.get("videos") or []
        return _expand_segment_video_placeholders(messages, videos)

    prompt = sample.get("prompt", "")
    response = sample.get("response", "")
    images = sample.get("images") or []
    videos = sample.get("videos") or []
    system_prompt = sample.get("system_prompt")
    video_placeholder_count = sum(_count_video_units(video) for video in videos)

    media_prefix = IMAGE_PLACEHOLDER + "\n" if len(images) == 1 else "".join(
        f"{IMAGE_PLACEHOLDER}\n" for _ in images
    )
    media_prefix += VIDEO_PLACEHOLDER + "\n" if video_placeholder_count == 1 else "".join(
        f"{VIDEO_PLACEHOLDER}\n" for _ in range(video_placeholder_count)
    )

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": media_prefix + prompt})
    messages.append({"role": "assistant", "content": response})
    return messages


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MossVLSFTDataset(Dataset):
    """Lightweight dataset that returns per-sample metadata.

    Heavy work (image loading, tokenisation, vision encoding) is deferred to
    :class:`MossVLDataCollator` so that the ``DataLoader`` workers only need
    to read JSON and build strings — no GPU / large-tensor serialisation.
    """

    def __init__(
        self,
        data_path: str,
        processor,
        data_dir: Optional[str] = None,
        max_length: int = 4096,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data: List[Dict[str, Any]] = json.load(f)
        if not isinstance(self.data, list):
            raise ValueError(f"Expected a JSON list in {data_path}")

        self.processor = processor
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(data_path))
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        messages = _build_messages(sample)
        text: str = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        labels_spans = find_assistant_spans(text)

        image_paths = [
            _resolve_path(self.data_dir, p)
            for p in (sample.get("images") or [])
        ]
        video_entries = [
            _resolve_path(self.data_dir, v)
            for v in (sample.get("videos") or [])
        ]

        return {
            "text": text,
            "image_paths": image_paths,
            "video_entries": video_entries,
            "labels_spans": labels_spans,
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class MossVLDataCollator:
    """Batch collator that calls the ``MossVLProcessor`` to produce
    model-ready tensors (``input_ids``, ``labels``, ``pixel_values``, …).

    Image loading and video decoding happen here (main process) so that
    ``DataLoader`` workers only return lightweight dicts.
    """

    def __init__(
        self,
        processor,
        max_length: Optional[int] = None,
        vision_chunked_length: int = 64,
    ):
        self.processor = processor
        self.max_length = max_length
        self.vision_chunked_length = vision_chunked_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [f["text"] for f in features]

        all_images: List[Image.Image] = []
        for f in features:
            for path in f["image_paths"]:
                all_images.append(Image.open(path).convert("RGB"))

        all_videos: list = []
        for f in features:
            all_videos.extend(f["video_entries"])

        all_spans = [f["labels_spans"] for f in features]

        kwargs: Dict[str, Any] = {
            "text": texts,
            "labels_spans": all_spans,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.max_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = self.max_length
        if all_images:
            kwargs["images"] = all_images
        if all_videos:
            kwargs["videos"] = all_videos

        batch = self.processor(**kwargs)

        batch["vision_chunked_length"] = self.vision_chunked_length
        return batch
