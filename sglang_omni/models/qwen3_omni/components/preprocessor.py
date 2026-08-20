# SPDX-License-Identifier: Apache-2.0
"""Model-specific preprocessor for Qwen3-Omni."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
import xxhash
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.action_timing import record_action_stage_timing
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import build_lightweight_mm_inputs
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing import (
    build_audio_mm_inputs,
    build_image_mm_inputs,
    build_video_mm_inputs,
    compute_audio_cache_key,
    compute_image_cache_key,
    compute_video_cache_key,
    ensure_audio_list_async,
    ensure_chat_template,
    ensure_image_list_async,
    ensure_video_list_async,
)
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload
from sglang_omni.utils.async_jsonl import enqueue_jsonl

logger = logging.getLogger(__name__)

_TRAIN_INPUT_TENSOR_NAMES = frozenset(
    {
        **build_image_mm_inputs({}),
        **build_audio_mm_inputs({}),
        **build_video_mm_inputs({}),
    }
)


def _resolve_local_model_dir(model_path: str) -> str:
    """Resolve a local model directory without eagerly hydrating full snapshots."""
    path = Path(model_path)
    if path.exists():
        return str(path)
    try:
        return str(resolve_model_path(model_path, local_files_only=True))
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "Local-only model resolution failed for %s; falling back to hub id",
            model_path,
            exc_info=exc,
        )
        return model_path


def _combine_cache_keys(*keys: str | None) -> str | None:
    parts = [key for key in keys if key]
    if not parts:
        return None
    return "|".join(parts)


# Special-token attributes the HF Qwen3OmniMoeProcessor reads off the tokenizer.
_QWEN3_OMNI_SPECIAL_TOKEN_KEYS = (
    "image_token",
    "audio_token",
    "video_token",
    "vision_bos_token",
    "vision_eos_token",
    "audio_bos_token",
    "audio_eos_token",
)


def _extra_special_tokens_compat(model_dir: str) -> dict[str, str]:
    """Rebuild ``extra_special_tokens`` for tokenizer_config exported by transformers 5.x.

    transformers 5.x writes the multimodal special tokens (``image_token`` etc.)
    as top-level keys in ``tokenizer_config.json`` instead of under the
    ``extra_special_tokens`` dict that transformers 4.x expects.
    """
    config_path = Path(model_dir) / "tokenizer_config.json"
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return {}
    if "extra_special_tokens" in config:
        return {}
    return {
        key: config[key]
        for key in _QWEN3_OMNI_SPECIAL_TOKEN_KEYS
        if isinstance(config.get(key), str)
    }


def _summarize_prompt_media(values: Any) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    if not isinstance(values, list):
        return summary
    for index, value in enumerate(values):
        if not isinstance(value, str):
            summary.append(
                {"index": index, "type": type(value).__name__, "repr": repr(value)}
            )
            continue
        digest = xxhash.xxh3_64_hexdigest(value.encode("utf-8"))
        if value.startswith("data:") and "," in value:
            header, encoded = value.split(",", 1)
            summary.append(
                {
                    "index": index,
                    "type": "data_uri",
                    "header": header,
                    "encoded_chars": len(encoded),
                    "xxhash": digest,
                }
            )
        else:
            summary.append(
                {
                    "index": index,
                    "type": "reference",
                    "value": value,
                    "chars": len(value),
                    "xxhash": digest,
                }
            )
    return summary


def _write_action_prompt_debug_record(record: dict[str, Any]) -> None:
    """Queue the rendered action prompt for postmortem analysis."""
    path = os.environ.get(
        "SGLANG_OMNI_ACTION_DEBUG_LOG_FILE", "/tmp/sglang-omni-action-debug.jsonl"
    )
    enqueue_jsonl(path, record)


def _contextualize_cache_key(base_key: str | None, **context: Any) -> str | None:
    if base_key is None:
        return None
    parts = [base_key]
    for key in sorted(context):
        value = context[key]
        if value is not None:
            parts.append(f"{key}={value}")
    return "|".join(parts)


DEFAULT_THINKER_MAX_NEW_TOKENS = 2048
QWEN3_OMNI_CHAT_TEMPLATE_FALLBACK_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def validate_prompt_seq_len(
    input_ids: torch.Tensor,
    *,
    max_seq_len: int | None,
    max_new_tokens: int = DEFAULT_THINKER_MAX_NEW_TOKENS,
    request_id: str | None = None,
) -> None:
    if max_seq_len is None:
        return
    prompt_len = int(input_ids.numel())
    if prompt_len >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"The input ({prompt_len} tokens) is longer than the model's "
            f"context length ({max_seq_len} tokens)."
        )
    total_tokens = prompt_len + int(max_new_tokens)
    if total_tokens >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} + "
            f"max_new_tokens {int(max_new_tokens)} = {total_tokens} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"Requested token count exceeds the model's maximum context length "
            f"of {max_seq_len} tokens. You requested a total of {total_tokens} "
            f"tokens: {prompt_len} tokens from the input messages and "
            f"{int(max_new_tokens)} tokens for the completion. Please reduce "
            f"the number of tokens in the input messages or the completion to "
            f"fit within the limit."
        )


def _is_pretokenized_prompt(inputs: Any) -> bool:
    """True when a rollout request carries pre-tokenized prompt ids.

    Miles RL rollout sends the exact prompt token ids it trains on, so those
    ids must bypass the chat template + HF processor to keep rollout and
    training tokens identical. A list of message dicts goes the normal path.
    """
    return (
        isinstance(inputs, list)
        and bool(inputs)
        and all(isinstance(token, int) for token in inputs)
    )


class Qwen3OmniPreprocessor:
    """CPU-side preprocessing and tokenization using the HF processor."""

    def __init__(
        self,
        model_path: str,
        max_seq_len: int | None = None,
        *,
        video_fps: float | None = None,
        video_max_frames: int | None = None,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
        video_total_pixels: int | None = None,
    ):
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.default_video_fps = float(video_fps) if video_fps is not None else None
        self.default_video_max_frames = (
            int(video_max_frames) if video_max_frames is not None else None
        )
        self.default_video_min_pixels = (
            int(video_min_pixels) if video_min_pixels is not None else None
        )
        self.default_video_max_pixels = (
            int(video_max_pixels) if video_max_pixels is not None else None
        )
        self.default_video_total_pixels = (
            int(video_total_pixels) if video_total_pixels is not None else None
        )
        self.model_dir = _resolve_local_model_dir(model_path)
        # Only override ``extra_special_tokens`` when the checkpoint omits them
        # (transformers 5.x layout). Passing an empty dict would clobber the
        # tokens a transformers 4.x checkpoint already declares in its config.
        extra_special_tokens = _extra_special_tokens_compat(self.model_dir)
        compat_kwargs = (
            {"extra_special_tokens": extra_special_tokens}
            if extra_special_tokens
            else {}
        )
        try:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
                **compat_kwargs,
            )
        except TypeError:
            if not compat_kwargs:
                raise
            logger.warning(
                "Qwen3OmniMoeProcessor.from_pretrained() rejected "
                "extra_special_tokens compat kwargs for %s; retrying without "
                "them",
                self.model_dir,
            )
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError, RuntimeError):
            if Path(model_path).exists():
                raise
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            self.model_dir = str(resolve_model_path(model_path, local_files_only=False))
        self.tokenizer = self.processor.tokenizer
        ensure_chat_template(
            self.tokenizer,
            model_path=self.model_dir,
            fallback_model_paths=(QWEN3_OMNI_CHAT_TEMPLATE_FALLBACK_MODEL,),
        )
        if not getattr(self.processor, "chat_template", None) and getattr(
            self.tokenizer, "chat_template", None
        ):
            self.processor.chat_template = self.tokenizer.chat_template
        # Category and child requests in one logical turn have identical
        # media/history. Keep only a small, one-shot cache of the prepared
        # media objects so child preprocessing does not decode them again.
        self._action_context_cache: dict[str, dict[str, Any]] = {}
        self._action_context_cache_lock = threading.Lock()
        self._action_context_cache_max_entries = 4

    def _action_context_cache_key(self, payload: StagePayload) -> str | None:
        metadata = getattr(payload.request, "metadata", None)
        if not isinstance(metadata, dict) or metadata.get("task") != "action_suffix_scoring":
            return None
        action_spec = payload.request.params.get("action_scoring")
        if not isinstance(action_spec, dict):
            return None
        key = action_spec.get("action_context_cache_key")
        stage = metadata.get("action_stage")
        if not isinstance(key, str) or not key.strip() or stage not in {"category", "child"}:
            return None
        return key.strip()

    def _get_action_context_cache(self, key: str | None) -> dict[str, Any] | None:
        if key is None:
            return None
        with self._action_context_cache_lock:
            # A context is consumed by the child stage exactly once. This
            # prevents a stale turn from retaining decoded media in memory.
            return self._action_context_cache.pop(key, None)

    def _put_action_context_cache(self, key: str | None, value: dict[str, Any]) -> None:
        if key is None:
            return
        with self._action_context_cache_lock:
            self._action_context_cache[key] = value
            while len(self._action_context_cache) > self._action_context_cache_max_entries:
                oldest = next(iter(self._action_context_cache))
                self._action_context_cache.pop(oldest, None)

    def _build_multimodal_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        num_images: int,
        num_audios: int,
        num_videos: int,
    ) -> list[dict[str, Any]]:
        """Convert simple messages to HF's structured multimodal format."""
        explicit = _count_explicit_mm(messages)
        if any(explicit.values()):
            expected = {
                "audio": num_audios,
                "image": num_images,
                "video": num_videos,
            }
            if explicit != expected:
                raise ValueError(
                    "Structured multimodal placeholders do not match the "
                    f"top-level media lists: placeholders={explicit}, "
                    f"media={expected}. Put media payloads in audios/images/"
                    "videos and keep one matching part in messages.content."
                )
            return messages
        if num_images == 0 and num_audios == 0 and num_videos == 0:
            return messages

        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Only inject placeholders into the last user message
            if i == len(messages) - 1 and role == "user":
                content_parts: list[dict[str, Any]] = []
                # Placeholders come BEFORE text (Qwen3-Omni format)
                for _ in range(num_images):
                    content_parts.append({"type": "image"})
                for _ in range(num_videos):
                    content_parts.append({"type": "video"})
                for _ in range(num_audios):
                    content_parts.append({"type": "audio"})
                content_parts.append({"type": "text", "text": content})
                result.append({"role": role, "content": content_parts})
            else:
                result.append(msg)

        return result

    async def __call__(self, payload: StagePayload) -> StagePayload:
        started = time.perf_counter()
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="preprocess_start",
        )
        try:
            result = await self._call_impl(payload)
            metadata = payload.request.metadata
            if (
                isinstance(metadata, dict)
                and metadata.get("task") == "action_suffix_scoring"
            ):
                state = Qwen3OmniPipelineState.from_dict(result.data)
                raw_inputs = state.raw_inputs if isinstance(state.raw_inputs, dict) else {}
                prompt = state.prompt if isinstance(state.prompt, dict) else {}
                input_ids = prompt.get("input_ids")
                encoder_cache_keys = {
                    stage: values.get("cache_key")
                    for stage, values in state.encoder_inputs.items()
                    if isinstance(values, dict) and values.get("cache_key")
                }
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                diagnostics = {
                    "event": "action_media_preprocess_completed",
                    "request_id": payload.request_id,
                    "session_id": metadata.get("session_id"),
                    "action_stage": metadata.get("action_stage"),
                    "logical_request_id": metadata.get("logical_request_id"),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "prompt_tokens": int(input_ids.numel())
                    if hasattr(input_ids, "numel")
                    else 0,
                    "audio_count": len(
                        raw_inputs.get("audios", raw_inputs.get("audio", [])) or []
                    ),
                    "image_count": len(raw_inputs.get("images", []) or []),
                    "encoder_cache_keys": encoder_cache_keys,
                    "action_context_cache_status": metadata.get(
                        "action_context_cache_status", "disabled"
                    ),
                }
                logger.info(
                    "action_media %s",
                    json.dumps(diagnostics, ensure_ascii=False, default=str),
                )
                record_action_stage_timing(
                    result,
                    "preprocessing",
                    wall_ms=round(elapsed_ms, 3),
                    prompt_tokens=diagnostics["prompt_tokens"],
                    audio_count=diagnostics["audio_count"],
                    image_count=diagnostics["image_count"],
                    context_cache_status=diagnostics[
                        "action_context_cache_status"
                    ],
                )
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="preprocess_end",
            )
        return result

    @staticmethod
    def _message_media_count(message: dict[str, Any], media_type: str) -> int:
        content = message.get("content")
        if not isinstance(content, list):
            return 0
        return sum(
            1
            for part in content
            if isinstance(part, dict) and part.get("type") == media_type
        )

    def _action_cache_metadata(
        self,
        payload: StagePayload,
        *,
        messages_mm: list[dict[str, Any]] | None,
        audios: list[Any],
        images: list[Any],
        input_ids: "torch.Tensor",
    ) -> dict[str, Any] | None:
        """Find the safe reusable boundary for an action-scoring prompt.

        The catalog system message and completed history precede the current
        turn.  Only that prefix is safe to reuse across turns; current media
        and the current avatar state must remain outside the reusable range.
        The boundary is measured from the exact processor output rather than
        reconstructed from text, because audio/image placeholders expand to
        model-specific token spans.
        """
        if not messages_mm:
            return None
        if payload.request.metadata.get("task") != "action_suffix_scoring":
            return None
        action_spec = payload.request.params.get("action_scoring")
        if not isinstance(action_spec, dict):
            return None
        history_count = action_spec.get("history_message_count")
        if not isinstance(history_count, int) or history_count < 0:
            return None

        system_count = int(messages_mm[0].get("role") == "system")
        boundary_end = system_count + history_count
        if boundary_end > len(messages_mm) or boundary_end == len(messages_mm):
            return None
        boundary_messages = messages_mm[:boundary_end]
        boundary_audio_count = sum(
            self._message_media_count(message, "audio")
            for message in boundary_messages
        )
        boundary_image_count = sum(
            self._message_media_count(message, "image")
            for message in boundary_messages
        )
        boundary_prompt = self.processor.apply_chat_template(
            boundary_messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        boundary_inputs = self.processor(
            text=boundary_prompt,
            images=images[:boundary_image_count] or None,
            audio=audios[:boundary_audio_count] or None,
            add_special_tokens=False,
            return_tensors="pt",
        )
        boundary_ids = boundary_inputs["input_ids"][0]
        boundary_len = int(boundary_ids.numel())
        if boundary_len <= 0 or boundary_len > int(input_ids.numel()):
            return None
        if not torch.equal(input_ids[:boundary_len].cpu(), boundary_ids.cpu()):
            logger.warning(
                "action cache boundary does not match full prompt; falling back "
                "to static catalog boundary request_id=%s boundary_tokens=%s full_tokens=%s",
                payload.request_id,
                boundary_len,
                int(input_ids.numel()),
            )
            return None
        return {
            "cache_prefix_token_count": boundary_len,
            "history_message_count": history_count,
            "history_audio_count": boundary_audio_count,
            "history_image_count": boundary_image_count,
        }

    def _finalize_state(
        self,
        payload: StagePayload,
        *,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        prompt_text: str,
        full_mm_inputs: dict[str, Any],
        encoder_inputs: dict[str, dict[str, Any]],
        action_cache_metadata: dict[str, Any] | None = None,
    ) -> StagePayload:
        """Assemble the thinker-ready pipeline state (single source of shape)."""
        request_inputs = payload.request.inputs
        raw_inputs = None
        if isinstance(request_inputs, dict):
            raw_inputs = {
                key: request_inputs[key]
                for key in ("audios", "audio", "images", "videos", "video", "audio_target_sr")
                if key in request_inputs
            }
        state = Qwen3OmniPipelineState(
            raw_inputs=raw_inputs,
            mm_inputs=build_lightweight_mm_inputs(full_mm_inputs),
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **(
                    {"action_scoring_cache": action_cache_metadata}
                    if action_cache_metadata is not None
                    else {}
                ),
            },
            encoder_inputs=encoder_inputs,
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        # Downstream projections consume the canonical state. Retaining request
        # inputs would duplicate raw media on every stage hop.
        payload.request.inputs = None
        for key in ("audios", "images", "videos"):
            payload.request.metadata.pop(key, None)
        return payload

    def _preprocess_train_inputs(
        self,
        payload: StagePayload,
        token_ids: list[int],
        bundle: dict[str, Any] | None = None,
    ) -> StagePayload:
        """Use Miles' exact token ids and optional processor tensors."""
        flat_inputs: dict[str, torch.Tensor] = {}
        processed_cache_key = None
        if bundle is not None:
            unknown_names = set(bundle["tensors"]) - _TRAIN_INPUT_TENSOR_NAMES
            if unknown_names:
                raise ValueError(
                    "unknown multimodal_train_inputs tensors: "
                    + ", ".join(sorted(unknown_names))
                )
            cache_parts = []
            for name in sorted(bundle["tensors"]):
                spec = bundle["tensors"][name]
                raw = base64.b64decode(spec["data"])
                cache_parts.append(
                    (
                        name,
                        spec["dtype"],
                        spec["shape"],
                        xxhash.xxh3_64_hexdigest(raw),
                    )
                )
                flat_inputs[name] = torch.frombuffer(
                    bytearray(raw),
                    dtype=getattr(torch, spec["dtype"]),
                ).reshape(spec["shape"])
            processed_cache_key = "processed:" + xxhash.xxh3_64_hexdigest(
                json.dumps(cache_parts, separators=(",", ":")).encode()
            )

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        validate_prompt_seq_len(
            input_ids,
            max_seq_len=self.max_seq_len,
            max_new_tokens=payload.request.params.get(
                "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
            ),
            request_id=payload.request_id,
        )

        full_mm_inputs: dict[str, Any] = {
            "image": build_image_mm_inputs(flat_inputs),
            "audio": build_audio_mm_inputs(flat_inputs),
            "video": build_video_mm_inputs(flat_inputs),
        }
        image_encoder_inputs = {
            name: value
            for name, value in {
                **full_mm_inputs["image"],
                **full_mm_inputs["video"],
            }.items()
            if value is not None
        }
        audio_encoder_inputs = {
            name: value
            for name, value in full_mm_inputs["audio"].items()
            if value is not None
        }
        has_image_payload = (
            image_encoder_inputs.get("pixel_values") is not None
            or image_encoder_inputs.get("pixel_values_videos") is not None
        )
        has_audio_payload = audio_encoder_inputs.get("input_features") is not None
        if image_encoder_inputs and not has_image_payload:
            raise ValueError(
                "multimodal_train_inputs provides image/video metadata "
                "without pixel_values or pixel_values_videos"
            )
        if audio_encoder_inputs and not has_audio_payload:
            raise ValueError(
                "multimodal_train_inputs provides audio metadata "
                "without input_features"
            )
        if processed_cache_key is not None:
            if image_encoder_inputs:
                image_encoder_inputs["cache_key"] = processed_cache_key
            if audio_encoder_inputs:
                audio_encoder_inputs["cache_key"] = processed_cache_key
        return self._finalize_state(
            payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_text="",
            full_mm_inputs=full_mm_inputs,
            encoder_inputs={
                "image_encoder": (
                    image_encoder_inputs
                    if has_image_payload
                    else {"_skip": True, "_result": {}}
                ),
                "audio_encoder": (
                    audio_encoder_inputs
                    if has_audio_payload
                    else {"_skip": True, "_result": {}}
                ),
            },
        )

    async def _call_impl(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if _is_pretokenized_prompt(inputs):
            return self._preprocess_train_inputs(payload, inputs)
        if isinstance(inputs, dict):
            multimodal_train_inputs = inputs.get("multimodal_train_inputs")
            if multimodal_train_inputs is not None:
                return self._preprocess_train_inputs(
                    payload,
                    inputs["input_ids"],
                    multimodal_train_inputs,
                )
            messages = inputs.get("messages", [])
            raw_images = inputs.get("images")
            raw_videos = inputs.get("videos") or inputs.get("video")
            raw_audios = inputs.get("audio") or inputs.get("audios")
            audio_target_sr = int(inputs.get("audio_target_sr", 16000))
            video_fps = inputs.get("video_fps", self.default_video_fps)
            video_max_frames = inputs.get(
                "video_max_frames",
                self.default_video_max_frames,
            )
            video_min_pixels = inputs.get(
                "video_min_pixels",
                self.default_video_min_pixels,
            )
            video_max_pixels = inputs.get(
                "video_max_pixels",
                self.default_video_max_pixels,
            )
            video_total_pixels = inputs.get(
                "video_total_pixels",
                self.default_video_total_pixels,
            )
            use_audio_in_video = inputs.get("use_audio_in_video")
            video_seconds_per_chunk = inputs.get("video_seconds_per_chunk")
            video_position_id_per_seconds = inputs.get("video_position_id_per_seconds")
            audio_from_video = False
            num_explicit_audios = 0
            resolved_video_fps = float(video_fps) if video_fps is not None else None
            resolved_video_max_frames = (
                int(video_max_frames) if video_max_frames is not None else None
            )
            resolved_video_min_pixels = (
                int(video_min_pixels) if video_min_pixels is not None else None
            )
            resolved_video_max_pixels = (
                int(video_max_pixels) if video_max_pixels is not None else None
            )
            resolved_video_total_pixels = (
                int(video_total_pixels) if video_total_pixels is not None else None
            )
            resolved_video_seconds_per_chunk = (
                float(video_seconds_per_chunk)
                if video_seconds_per_chunk is not None
                else None
            )
            resolved_video_position_id_per_seconds = (
                float(video_position_id_per_seconds)
                if video_position_id_per_seconds is not None
                else None
            )

            # Compute cache keys BEFORE conversion (paths are cheap to hash)
            image_cache_key = compute_image_cache_key(raw_images)
            raw_audio_cache_key = compute_audio_cache_key(raw_audios)
            video_cache_key = compute_video_cache_key(raw_videos)
            action_context_key = self._action_context_cache_key(payload)
            cached_context = self._get_action_context_cache(
                action_context_key
                if payload.request.metadata.get("action_stage") == "child"
                else None
            )
            action_context_cache_status = "hit" if cached_context is not None else "miss"

            # Count explicit audio inputs (for placeholder insertion)
            if raw_audios:
                num_explicit_audios = (
                    len(raw_audios) if isinstance(raw_audios, list) else 1
                )

            # Reuse the category stage's loaded media for the child stage.
            # The child still renders its own text prompt, while media loading
            # and decoding happen only once for this logical turn.
            if cached_context is not None:
                images = cached_context["images"]
                videos = cached_context["videos"]
                audios = cached_context["audios"]
                sampled_video_fps = cached_context["sampled_video_fps"]
                extracted_audio_from_video = []
                audio_from_video = bool(cached_context.get("audio_from_video"))
                audios_result = audios
            else:
                # If we need audio from video, extract it during video loading
                # to avoid duplicate downloads.
                extract_audio_from_video_flag = bool(use_audio_in_video and raw_videos)
                images, videos_result, audios_result = await asyncio.gather(
                    ensure_image_list_async(raw_images),
                    ensure_video_list_async(
                        raw_videos,
                        fps=resolved_video_fps,
                        max_frames=resolved_video_max_frames,
                        min_pixels=resolved_video_min_pixels,
                        max_pixels=resolved_video_max_pixels,
                        total_pixels=resolved_video_total_pixels,
                        extract_audio=extract_audio_from_video_flag,
                        audio_target_sr=audio_target_sr,
                    ),
                    ensure_audio_list_async(raw_audios, target_sr=audio_target_sr),
                )
                videos, sampled_video_fps, extracted_audio_from_video = videos_result

            # Merge extracted audio from videos with explicit audio (if any)
            if extracted_audio_from_video:
                # Filter out None values (videos without audio)
                extracted_audio_from_video = [
                    audio for audio in extracted_audio_from_video if audio is not None
                ]
                if extracted_audio_from_video:
                    audio_from_video = True
                    # Merge with explicit audio
                    if audios_result:
                        if isinstance(audios_result, list):
                            audios = audios_result + extracted_audio_from_video
                        else:
                            audios = [audios_result] + extracted_audio_from_video
                    else:
                        audios = extracted_audio_from_video
                else:
                    audios = audios_result
            else:
                audios = audios_result

            if (
                cached_context is None
                and action_context_key is not None
                and payload.request.metadata.get("action_stage") == "category"
            ):
                self._put_action_context_cache(
                    action_context_key,
                    {
                        "images": images,
                        "videos": videos,
                        "audios": audios,
                        "sampled_video_fps": sampled_video_fps,
                        "audio_from_video": audio_from_video,
                    },
                )
            payload.request.metadata["action_context_cache_status"] = (
                action_context_cache_status
            )
        else:
            action_context_cache_status = "disabled"
            messages = inputs
            images = []
            videos = []
            audios = []
            image_cache_key = None
            raw_audio_cache_key = None
            video_cache_key = None
            audio_target_sr = 16000
            video_fps = self.default_video_fps
            video_max_frames = self.default_video_max_frames
            video_min_pixels = self.default_video_min_pixels
            video_max_pixels = self.default_video_max_pixels
            video_total_pixels = self.default_video_total_pixels
            sampled_video_fps = None
            use_audio_in_video = None
            video_seconds_per_chunk = None
            video_position_id_per_seconds = None
            audio_from_video = False
            num_explicit_audios = 0
            resolved_video_fps = None
            resolved_video_max_frames = None
            resolved_video_min_pixels = None
            resolved_video_max_pixels = None
            resolved_video_total_pixels = None
            resolved_video_seconds_per_chunk = None
            resolved_video_position_id_per_seconds = None

        messages_norm = _official_normalize_messages(messages)
        # Insert placeholders:
        # - Explicit audio files get independent audio placeholders
        # - Video audio (when use_audio_in_video=True) is handled by video token, no separate placeholder
        num_audios_for_placeholder = num_explicit_audios
        messages_mm = self._build_multimodal_messages(
            messages_norm,
            num_images=len(images),
            num_audios=num_audios_for_placeholder,
            num_videos=len(videos),
        )
        prompt_text = self.processor.apply_chat_template(
            messages_mm,
            add_generation_prompt=True,
            tokenize=False,
        )

        videos_kwargs: dict[str, Any] = {}
        if sampled_video_fps is not None:
            videos_kwargs["fps"] = (
                sampled_video_fps[0]
                if len(sampled_video_fps) == 1
                else sampled_video_fps
            )
        elif resolved_video_fps is not None:
            videos_kwargs["fps"] = resolved_video_fps
        if resolved_video_max_frames is not None:
            videos_kwargs["max_frames"] = resolved_video_max_frames
        if resolved_video_min_pixels is not None:
            videos_kwargs["min_pixels"] = resolved_video_min_pixels
        if resolved_video_max_pixels is not None:
            videos_kwargs["max_pixels"] = resolved_video_max_pixels
        if resolved_video_total_pixels is not None:
            videos_kwargs["total_pixels"] = resolved_video_total_pixels
        if use_audio_in_video is not None:
            videos_kwargs["use_audio_in_video"] = bool(use_audio_in_video)
        if resolved_video_seconds_per_chunk is not None:
            videos_kwargs["seconds_per_chunk"] = resolved_video_seconds_per_chunk
        if resolved_video_position_id_per_seconds is not None:
            videos_kwargs["position_id_per_seconds"] = float(
                resolved_video_position_id_per_seconds
            )
        if videos:
            # torchcodec backend expects a non-None device string
            videos_kwargs.setdefault("device", "cpu")
        processor_kwargs: dict[str, Any] = {}
        if videos_kwargs:
            processor_kwargs["videos_kwargs"] = videos_kwargs

        hf_inputs = self.processor(
            text=prompt_text,
            images=images or None,
            videos=videos or None,
            audio=audios or None,
            add_special_tokens=False,
            return_tensors="pt",
            **processor_kwargs,
        )

        input_ids = hf_inputs["input_ids"][0]
        if payload.request.metadata.get("task") == "action_suffix_scoring":
            prompt_diagnostics = {
                "event": "action_scoring_prompt_rendered",
                "timestamp_unix_ms": round(time.time() * 1000.0),
                "request_id": payload.request_id,
                "session_id": payload.request.metadata.get("session_id"),
                "prompt_tokens": int(input_ids.numel()),
                "full_prompt": prompt_text,
                "messages": (payload.request.inputs or {}).get("messages", []) if isinstance(payload.request.inputs, dict) else [],
                "audios": _summarize_prompt_media((payload.request.inputs or {}).get("audios", []) if isinstance(payload.request.inputs, dict) else []),
                "images": _summarize_prompt_media((payload.request.inputs or {}).get("images", []) if isinstance(payload.request.inputs, dict) else []),
                "params": payload.request.params,
                "metadata": payload.request.metadata,
            }
            logger.info(
                "Qwen3-Omni action scoring prompt rendered=%s",
                json.dumps(prompt_diagnostics, ensure_ascii=False, default=str),
            )
            _write_action_prompt_debug_record(prompt_diagnostics)
        attention_mask = hf_inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask[0]
        else:
            attention_mask = torch.ones_like(input_ids)

        try:
            validate_prompt_seq_len(
                input_ids,
                max_seq_len=self.max_seq_len,
                max_new_tokens=payload.request.params.get(
                    "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
                ),
                request_id=payload.request_id,
            )
        except ValueError:
            request_inputs = payload.request.inputs
            request_inputs = request_inputs if isinstance(request_inputs, dict) else {}
            diagnostics = {
                "request_id": payload.request_id,
                "prompt_tokens": int(input_ids.numel()),
                "max_seq_len": self.max_seq_len,
                "max_new_tokens": payload.request.params.get(
                    "max_new_tokens", DEFAULT_THINKER_MAX_NEW_TOKENS
                ),
                "full_prompt": prompt_text,
                "messages": request_inputs.get("messages", []),
                "audios": _summarize_prompt_media(request_inputs.get("audios", [])),
                "images": _summarize_prompt_media(request_inputs.get("images", [])),
                "videos": _summarize_prompt_media(request_inputs.get("videos", [])),
                "params": payload.request.params,
                "metadata": payload.request.metadata,
            }
            logger.error(
                "Qwen3-Omni prompt validation failed; full prompt diagnostics=%s",
                json.dumps(diagnostics, ensure_ascii=False, default=str),
                exc_info=True,
            )
            raise

        full_mm_inputs: dict[str, Any] = {
            "image": build_image_mm_inputs(hf_inputs),
            "audio": build_audio_mm_inputs(hf_inputs),
            "video": build_video_mm_inputs(hf_inputs),
        }
        if use_audio_in_video is not None:
            full_mm_inputs["video"]["use_audio_in_video"] = bool(use_audio_in_video)

        # Build encoder_inputs with cache_key for efficient caching.
        # Include preprocessing parameters that materially change encoder outputs.
        image_encoder_inputs = {
            **full_mm_inputs["image"],
            **full_mm_inputs["video"],
        }
        effective_video_fps: tuple[float, ...] | None = None
        if sampled_video_fps is not None:
            effective_video_fps = tuple(float(fps) for fps in sampled_video_fps)
        elif resolved_video_fps is not None:
            effective_video_fps = (resolved_video_fps,)

        contextual_video_cache_key = _contextualize_cache_key(
            video_cache_key,
            fps=effective_video_fps,
            max_frames=resolved_video_max_frames,
            min_pixels=resolved_video_min_pixels,
            max_pixels=resolved_video_max_pixels,
            total_pixels=resolved_video_total_pixels,
            seconds_per_chunk=resolved_video_seconds_per_chunk,
        )
        combined_cache_key = _combine_cache_keys(
            image_cache_key, contextual_video_cache_key
        )
        if combined_cache_key:
            image_encoder_inputs["cache_key"] = combined_cache_key

        audio_encoder_inputs = {**full_mm_inputs["audio"]}
        contextualized_audio_cache_key = _contextualize_cache_key(
            raw_audio_cache_key,
            target_sr=audio_target_sr,
        )
        if audio_from_video:
            contextualized_audio_cache_key = _combine_cache_keys(
                contextualized_audio_cache_key,
                _contextualize_cache_key(
                    video_cache_key,
                    extracted_audio=True,
                    target_sr=audio_target_sr,
                ),
            )
        if contextualized_audio_cache_key:
            audio_encoder_inputs["cache_key"] = contextualized_audio_cache_key

        encoder_inputs: dict[str, dict[str, Any]] = {}
        image_encoder_inputs = {
            k: v for k, v in image_encoder_inputs.items() if v is not None
        }
        if (
            image_encoder_inputs.get("pixel_values") is not None
            or image_encoder_inputs.get("pixel_values_videos") is not None
        ):
            encoder_inputs["image_encoder"] = image_encoder_inputs
        else:
            encoder_inputs["image_encoder"] = {"_skip": True, "_result": {}}
        if audio_encoder_inputs.get("input_features") is not None:
            encoder_inputs["audio_encoder"] = audio_encoder_inputs
        else:
            encoder_inputs["audio_encoder"] = {"_skip": True, "_result": {}}

        action_cache_metadata = self._action_cache_metadata(
            payload,
            messages_mm=messages_mm,
            audios=audios,
            images=images,
            input_ids=input_ids,
        )
        return self._finalize_state(
            payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_text=prompt_text,
            full_mm_inputs=full_mm_inputs,
            encoder_inputs=encoder_inputs,
            action_cache_metadata=action_cache_metadata,
        )
# Official OpenAI/Qwen messages may carry multimodal parts in each content list.


_MM_TYPE_ALIASES = {
    "audio": "audio",
    "input_audio": "audio",
    "audio_url": "audio",
    "image": "image",
    "image_url": "image",
    "video": "video",
    "video_url": "video",
}
_MM_PART_TYPES = ("audio", "image", "video")


def _normalize_content_parts(content: Any) -> list[dict[str, Any]] | None:
    """Normalize recognized structured multimodal content to HF parts."""
    if not isinstance(content, list) or not content:
        return None

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        if item_type == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif isinstance(item_type, str) and item_type in _MM_TYPE_ALIASES:
            parts.append({"type": _MM_TYPE_ALIASES[item_type]})
        else:
            return None
    return parts


def _official_normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalize messages without flattening recognized structured content."""
    if not isinstance(messages, list):
        raise ValueError("Preprocessing expects a list of chat messages")

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a dict with role/content")
        role = message.get("role", "user")
        content = message.get("content", "")
        parts = _normalize_content_parts(content)
        if parts is not None:
            normalized.append({"role": role, "content": parts})
        elif isinstance(content, str):
            normalized.append({"role": role, "content": content})
        else:
            normalized.append(
                {"role": role, "content": json.dumps(content, ensure_ascii=True)}
            )
    return normalized


def _count_explicit_mm(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Count multimodal parts already positioned by the caller."""
    counts = {part_type: 0 for part_type in _MM_PART_TYPES}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            item_type = item.get("type") if isinstance(item, dict) else None
            if isinstance(item_type, str) and item_type in counts:
                counts[item_type] += 1
    return counts
