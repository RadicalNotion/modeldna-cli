"""
Known base model reference signatures.
Each entry anchors a canonical architecture family by its config.json fingerprint.
"""
from __future__ import annotations

KNOWN_BASES: dict[str, dict] = {
    "qwen3_5_text": {
        "name": "Qwen3.5 (dense)",
        "vocab_size": 248320,
        "model_type_patterns": ["qwen3_5_text", "qwen3_5"],
    },
    "qwen3_5_moe_text": {
        "name": "Qwen3.5 MoE",
        "vocab_size": 248320,
        "model_type_patterns": ["qwen3_5_moe_text", "qwen3_5_moe"],
    },
    "qwen3": {
        "name": "Qwen3",
        "vocab_size": [151936, 152064],
        "model_type_patterns": ["qwen3"],
    },
    "qwen2": {
        "name": "Qwen2.5 (incl. VL)",
        "vocab_size": [151936, 152064, 151680],
        "model_type_patterns": ["qwen2"],
    },
    "llama3": {
        "name": "Llama 3.x",
        "vocab_size": 128256,
        "model_type_patterns": ["llama"],
        "num_key_value_heads_hint": [8, 32],
    },
    "llama2": {
        "name": "Llama 2",
        "vocab_size": 32000,
        "model_type_patterns": ["llama"],
    },
    "mistral": {
        "name": "Mistral 7B family",
        "vocab_size": 32000,
        "model_type_patterns": ["mistral", "mixtral"],
    },
    "deepseek_v3": {
        "name": "DeepSeek V3/R1",
        "vocab_size": 129280,
        "model_type_patterns": ["deepseek_v3", "deepseek_v2"],
        "kv_lora_rank": 512,
    },
    "gemma": {
        "name": "Gemma family",
        "vocab_size": [256000, 262144],
        "model_type_patterns": ["gemma"],
    },
    "glm4": {
        "name": "ZhipuAI GLM-4.x (4.5 / 4.6 / 4.7 / 4.6V)",
        "vocab_size": [151552, 151936, 154880],
        "model_type_patterns": ["glm4v_moe_text", "glm4v_moe", "glm4_moe_lite", "glm4_moe", "glm4", "chatglm"],
    },
    "internlm3": {
        "name": "InternLM 3",
        "vocab_size": 92544,
        "model_type_patterns": ["internlm2", "internlm3"],
    },
    "phi4": {
        "name": "Phi-4 family",
        "vocab_size": [100352, 100352],
        "model_type_patterns": ["phi3", "phi4"],
    },
    "nemotron_h": {
        "name": "NemotronH (NVIDIA Mamba+MoE hybrid)",
        "vocab_size": 131072,
        "model_type_patterns": ["nemotron_h", "nemotronh"],
    },
    "seed_oss": {
        "name": "ByteDance Seed-OSS (dense)",
        "vocab_size": 155136,
        "model_type_patterns": ["seed_oss"],
    },
    "bailing_v2": {
        "name": "AntGroup Bailing-V2 / V2.5 (inclusionAI Ling)",
        "vocab_size": 157184,
        "model_type_patterns": ["bailing_hybrid", "bailing_moe", "bailingmm_moe_v2_lite"],
    },
    "llada2": {
        "name": "inclusionAI LLaDA2 (discrete-diffusion MoE)",
        "vocab_size": [157184, 173568],
        "model_type_patterns": ["llada2_moe", "llada2"],
        # 157184=text, 173568=Uni any-to-any (image codebook merged into vocab)
    },
    "kimi": {
        "name": "Moonshot Kimi (K2, Kimi-Linear)",
        "vocab_size": 163840,
        "model_type_patterns": ["kimi_linear", "kimi"],
    },
    "ernie4_5_vl": {
        "name": "Baidu ERNIE 4.5 VL (MoE multimodal)",
        "vocab_size": 103424,
        "model_type_patterns": ["ernie4_5_moe_vl", "ernie4_5_vl"],
    },
    "qianfan_vl": {
        "name": "Baidu Qianfan-VL (dense multimodal)",
        "vocab_size": 182025,
        "model_type_patterns": ["qianfanvl_chat", "qianfan"],
    },
    "interns1": {
        "name": "InternLM S1 (dense, long-chain reasoning)",
        "vocab_size": 153216,
        "model_type_patterns": ["interns1"],
    },
    "pangu_pro_moe": {
        "name": "FreedomIntelligence Pangu-R (Huawei Pangu-Pro-MoE)",
        "vocab_size": 153600,
        "model_type_patterns": ["pangupromoe"],
    },
    "iquest_coder": {
        "name": "IQuest-Coder",
        "vocab_size": 76800,
        "model_type_patterns": ["iquestcoder"],
    },
    "minicpm": {
        "name": "OpenBMB MiniCPM",
        "vocab_size": 73448,
        "model_type_patterns": ["minicpm"],
    },
    "step3_5": {
        "name": "StepFun Step-3.5 Flash",
        "vocab_size": [128815, 128896],
        "model_type_patterns": ["step3p5"],
    },
    "mimo_v2": {
        "name": "Xiaomi MiMo V2.x",
        "vocab_size": 152576,
        "model_type_patterns": ["mimo_v2"],
    },
    "hunyuan_v1":
        {"name": "Tencent Hunyuan V1 (dense + MoT multimodal)",
        "vocab_size": 120818,
        "model_type_patterns": ["hunyuan_v1_dense", "hunyuan_vl_mot", "hunyuan"],
    },
    "gpt_oss": {
        "name": "OpenAI gpt-oss (via InternVL3.5 wrapper)",
        "vocab_size": 200028,
        "model_type_patterns": ["gpt_oss"],
    },
    "valley": {
        "name": "ByteDance Valley (video-language)",
        "vocab_size": [151675, 151679],
        "model_type_patterns": ["valley"],
    },
    "emu3": {
        "name": "BAAI Emu3 family (unified vision+text)",
        "vocab_size": [184622, 282926],
        "model_type_patterns": ["emu3"],
    },
}
