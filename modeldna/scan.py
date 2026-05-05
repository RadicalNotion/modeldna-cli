"""
modeldna Stage 1 scanner — architecture fingerprinting from config.json only.
No weight download required. Completes in ~2 seconds.
"""
from __future__ import annotations
import hashlib, time
from datetime import datetime, timezone
from typing import Optional
import requests

from .bases import KNOWN_BASES

HF_API = "https://huggingface.co"


def fetch_config(model_id: str) -> Optional[dict]:
    url = f"{HF_API}/{model_id}/resolve/main/config.json"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_metadata(model_id: str) -> dict:
    try:
        r = requests.get(f"{HF_API}/api/models/{model_id}", timeout=10)
        r.raise_for_status()
        d = r.json()
        return {
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "author": d.get("author", ""),
            "tags": d.get("tags", []),
            "pipeline_tag": d.get("pipeline_tag", ""),
            "base_model": d.get("cardData", {}).get("base_model", ""),
            "license": d.get("cardData", {}).get("license", ""),
            "created_at": d.get("createdAt", ""),
            "last_modified": d.get("lastModified", ""),
        }
    except Exception:
        return {}


def detect_claims(model_id: str, metadata: dict) -> dict:
    """Detect what base model the name/metadata claims."""
    claims: dict = {}
    name = model_id.split("/")[-1].lower()

    if metadata.get("base_model"):
        claims["explicit_base"] = metadata["base_model"]

    name_signals = []
    for term, base_key in [
        ("qwen3.5", "qwen3_5"), ("qwen3-5", "qwen3_5"), ("qwen35", "qwen3_5"),
        ("qwen3", "qwen3"), ("qwen2.5", "qwen2"), ("qwen2", "qwen2"),
        ("llama-3", "llama3"), ("llama3", "llama3"), ("llama-2", "llama2"),
        ("mistral", "mistral"), ("mixtral", "mistral"),
        ("deepseek", "deepseek_v3"), ("gemma", "gemma"),
        ("glm", "glm4"), ("internlm", "internlm3"), ("phi", "phi4"),
    ]:
        if term in name:
            name_signals.append(base_key)
    if name_signals:
        claims["name_implies"] = name_signals

    suspicious = [t for t in ["claude", "gpt", "chatgpt", "openai", "gemini", "anthropic"] if t in name]
    if suspicious:
        claims["suspicious_name_terms"] = suspicious

    return claims


def stage1_screen(config: dict) -> dict:
    """Match config fingerprint against KNOWN_BASES. Handles nested text_config."""
    # Merge text_config (multimodal pattern: Qwen3.5, Mistral3, MiMo-V2.5)
    if config.get("text_config") and not config.get("vocab_size"):
        tc = config["text_config"]
        config = {**tc, **{k: v for k, v in config.items()
                           if k not in ("text_config", "vision_config", "audio_config")}}

    vocab = config.get("vocab_size")
    model_type = (config.get("model_type") or "").lower()
    hidden = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    kv_lora = config.get("kv_lora_rank")

    key_fields = sorted([f"vocab={vocab}", f"type={model_type}", f"hidden={hidden}",
                         f"layers={layers}", f"kv_lora={kv_lora}"])
    arch_sig = hashlib.md5("|".join(str(f) for f in key_fields).encode()).hexdigest()[:12]

    base_matches = []
    for base_key, base_info in KNOWN_BASES.items():
        score, reasons = 0, []
        expected_vocab = base_info.get("vocab_size")
        if isinstance(expected_vocab, list):
            if vocab in expected_vocab:
                score += 3; reasons.append(f"vocab matches ({vocab})")
        elif vocab == expected_vocab:
            score += 3; reasons.append(f"vocab matches ({vocab})")
        for pat in base_info.get("model_type_patterns", []):
            if model_type == pat:
                score += 3; reasons.append(f"model_type '{model_type}' exact"); break
            elif model_type.startswith(pat):
                score += 2; reasons.append(f"model_type '{model_type}' ~ {pat}"); break
        if base_key == "deepseek_v3" and kv_lora and kv_lora > 0:
            score += 2; reasons.append(f"MLA kv_lora_rank={kv_lora}")
        if score >= 3:
            base_matches.append({
                "base": base_key,
                "name": base_info["name"],
                "confidence": "HIGH" if score >= 5 else "MODERATE",
                "score": score,
                "evidence": reasons,
            })

    return {
        "arch_signature": arch_sig,
        "config_signals": {
            "model_type": model_type, "vocab_size": vocab,
            "hidden_size": hidden, "num_layers": layers,
            "has_mla": bool(kv_lora and kv_lora > 0), "kv_lora_rank": kv_lora,
        },
        "base_matches": sorted(base_matches, key=lambda x: -x["score"]),
    }


def generate_verdict(model_id: str, metadata: dict, claims: dict, stage1: dict) -> dict:
    base_matches = stage1["base_matches"]
    suspicious = claims.get("suspicious_name_terms", [])

    if base_matches:
        top = base_matches[0]
        arch_verdict = (f"CONFIRMED — architecture matches {top['name']}"
                        if top["confidence"] == "HIGH"
                        else f"LIKELY — architecture consistent with {top['name']}")
    else:
        arch_verdict = "UNRECOGNIZED — architecture does not match any known base model"

    flags = []
    if "claude" in suspicious or "anthropic" in suspicious:
        flags.append({
            "type": "UNVERIFIABLE_CLAIM", "term": "claude/anthropic",
            "explanation": (
                "Claude weights are not publicly available — no weight transfer from Claude "
                "is possible. If this model used Claude-generated reasoning traces (distillation), "
                "that is a post-training technique that leaves no architectural trace and cannot "
                "be verified from weights alone."
            ),
        })
    if "gpt" in suspicious or "openai" in suspicious or "chatgpt" in suspicious:
        flags.append({
            "type": "UNVERIFIABLE_CLAIM", "term": "gpt/openai",
            "explanation": "GPT-4/OpenAI weights are closed. Any weight transfer claim is false. "
                           "Distillation via outputs is possible but unverifiable from architecture.",
        })
    if "gemini" in suspicious:
        flags.append({
            "type": "UNVERIFIABLE_CLAIM", "term": "gemini",
            "explanation": "Gemini weights are closed. Architecture shows no Gemini structure.",
        })

    name_implied = claims.get("name_implies", [])
    if name_implied and base_matches:
        top_base = base_matches[0]["base"]
        if not any(n in top_base or top_base in n for n in name_implied):
            flags.append({
                "type": "NAME_MISMATCH",
                "explanation": f"Name implies {name_implied} but architecture suggests {top_base}.",
            })

    return {
        "model_id": model_id,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "verdict": {
            "architecture": arch_verdict,
            "base_model_confirmed": base_matches[0]["name"] if base_matches else "Unknown",
            "confidence": base_matches[0]["confidence"] if base_matches else "NONE",
            "flags": flags,
            "flag_count": len(flags),
            "stage": "Stage 1 (config-only — no weight download)",
        },
        "evidence": {
            "config_signals": stage1["config_signals"],
            "base_matches": stage1["base_matches"][:3],
            "claimed_base": claims.get("explicit_base"),
            "name_implies": name_implied,
        },
        "metadata": {
            "downloads": metadata.get("downloads", 0),
            "likes": metadata.get("likes", 0),
            "license": metadata.get("license", ""),
            "created_at": metadata.get("created_at", ""),
        },
    }


def scan(model_id: str) -> dict:
    """Stage 1 scan. Primary entry point."""
    # Normalize HF URLs
    if "huggingface.co/" in model_id:
        model_id = model_id.split("huggingface.co/")[-1].strip("/")

    t0 = time.time()

    # Detect unsupported formats before attempting config fetch
    if "gguf" in model_id.lower():
        return {
            "model_id": model_id,
            "error": (
                "GGUF models pack weights into a single file and don't have a standard config.json. "
                "Stage 1 scanning works with standard HuggingFace checkpoints (safetensors/PyTorch). "
                "Try the original non-quantized model instead. GGUF support is on the roadmap."
            ),
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }

    config = fetch_config(model_id)
    if not config:
        return {
            "model_id": model_id,
            "error": "Could not fetch config.json — model may be private, gated, or not exist on HuggingFace.",
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }
    metadata = fetch_metadata(model_id)
    claims = detect_claims(model_id, metadata)
    stage1 = stage1_screen(config)
    result = generate_verdict(model_id, metadata, claims, stage1)
    result["elapsed_s"] = round(time.time() - t0, 2)
    return result
