"""Unit tests for Stage 1 scanner — no network calls."""
from modeldna.scan import stage1_screen, detect_claims, generate_verdict


QWEN35_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "vocab_size": 248320,
    "hidden_size": 2048,
    "num_hidden_layers": 94,
}

NESTED_QWEN35_CONFIG = {
    "model_type": "qwen3_5_vl",
    "text_config": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
    },
    "vision_config": {"model_type": "clip"},
}

DEEPSEEK_CONFIG = {
    "model_type": "deepseek_v3",
    "vocab_size": 129280,
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "kv_lora_rank": 512,
}


def test_qwen35_moe_identified():
    result = stage1_screen(QWEN35_CONFIG)
    assert result["base_matches"], "Should match at least one base"
    assert result["base_matches"][0]["base"] == "qwen3_5_moe_text"
    assert result["base_matches"][0]["confidence"] == "HIGH"


def test_nested_text_config_merged():
    result = stage1_screen(NESTED_QWEN35_CONFIG)
    assert result["config_signals"]["vocab_size"] == 248320, "Should lift vocab from text_config"
    assert result["base_matches"], "Should still identify base after merge"


def test_deepseek_mla_signal():
    result = stage1_screen(DEEPSEEK_CONFIG)
    assert result["config_signals"]["has_mla"] is True
    assert result["base_matches"][0]["base"] == "deepseek_v3"
    assert result["base_matches"][0]["confidence"] == "HIGH"


def test_claude_flag_raised():
    claims = detect_claims("user/Qwen3.5-Claude-4.6-Distilled", {})
    assert "claude" in claims.get("suspicious_name_terms", [])


def test_no_false_flags_clean_model():
    claims = detect_claims("Qwen/Qwen3.5-27B", {})
    assert "suspicious_name_terms" not in claims
