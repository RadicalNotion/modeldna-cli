# 🧬 modeldna-cli

**The DNA test for AI models — verify provenance before you download.**

```
modeldna scan Qwen/Qwen3.5-27B
modeldna scan Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled
```

Paste any HuggingFace model ID or URL. Get an instant architecture fingerprint, claim validation, and flagged inconsistencies — with no weight download.

---

## What it checks (Stage 1 — free, unlimited)

| Check | What it does |
|---|---|
| **Architecture confirmation** | Matches `config.json` fingerprint against 12+ known base families |
| **Claim validation** | Does the model name match the actual architecture? |
| **Unverifiable claim flags** | "Claude-distilled", "GPT-based" — warns when a claim can't be verified from weights |
| **Name vs architecture mismatch** | Flags models whose name implies one base but whose config says another |

Stage 1 uses only `config.json` (~2 KB). No weights downloaded. Results in ~2 seconds.

> Stage 2 (weight-level analysis) — coming soon. Requires model download; provides stronger confirmation via embedding alignment and vocabulary weight correlation.

---

## Install

```bash
pip install modeldna
```

Requires Python 3.10+. No API key needed for Stage 1.

---

## Usage

```bash
# Rich terminal output (default)
modeldna scan Qwen/Qwen3.5-27B

# Raw JSON
modeldna scan deepseek-ai/DeepSeek-R1 --json
```

### Python API

```python
from modeldna.scan import scan

result = scan("Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled")
print(result["verdict"]["architecture"])
# CONFIRMED — architecture matches Qwen3.5 MoE

print(result["verdict"]["flags"])
# [{"type": "UNVERIFIABLE_CLAIM", "term": "claude/anthropic", "explanation": "..."}]
```

---

## Example output

```
╭─ 🧬 Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled ─╮
│ CONFIRMED — architecture matches Qwen3.5 MoE                        │
│ Confidence: HIGH  |  📥 30,419 downloads  |  Scanned in 0.32s      │
╰─────────────────────────────────────────────────────────────────────╯

 Base              Confidence   Score   Evidence
 Qwen3.5 MoE       HIGH         5       vocab matches (248320); model_type 'qwen3_5_moe_text' ~ qwen3_5_moe

╭─ ⚠ UNVERIFIABLE_CLAIM ──────────────────────────────────────────────╮
│ Claude weights are not publicly available — no weight transfer from  │
│ Claude is possible. If this model used Claude-generated reasoning    │
│ traces (distillation), that is a post-training technique that leaves │
│ no architectural trace and cannot be verified from weights alone.    │
╰─────────────────────────────────────────────────────────────────────╯
```

---

## Architecture

```
modeldna/
├── bases.py     # KNOWN_BASES: canonical fingerprints for 12 base families
├── scan.py      # Stage 1 scanner: fetch → fingerprint → match → verdict
└── cli.py       # CLI entry point (rich output + --json)
```

The scanner is intentionally dependency-light: `requests` + `rich` only. No ML framework required.

---

## How it works

1. **Fetch** `config.json` from HuggingFace (~2 KB, no auth needed for public models)
2. **Fingerprint** the architecture: `vocab_size`, `model_type`, `hidden_size`, `kv_lora_rank`
3. **Match** against the `KNOWN_BASES` reference table (scored by vocab + model_type + special signals like MLA)
4. **Validate claims** in the model name against the confirmed architecture
5. **Return** a structured verdict with evidence and flags

Powered by [ModelAtlas](https://modeldna.ai) — the private architecture intelligence corpus that backs the reference data.

---

## Contributing

The `KNOWN_BASES` dictionary in `modeldna/bases.py` is the primary contribution target. If you find a model family not yet recognized, open a PR with the `vocab_size`, `model_type_patterns`, and a real HF model ID to test against.

---

*modeldna.ai · a [RadicalNotion](https://radicalnotion.ai) product · Apache-2.0*
