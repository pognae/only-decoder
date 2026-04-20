import yaml
import os
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
}

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_model_from_config(config_path: str):
    cfg = load_yaml(config_path)
    m = cfg["model"]
    config = LlamaConfig(
        vocab_size=m["vocab_size"],
        hidden_size=m["hidden_size"],
        intermediate_size=m["intermediate_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        num_key_value_heads=m["num_key_value_heads"],
        max_position_embeddings=m["max_position_embeddings"],
        bos_token_id=m["bos_token_id"],
        eos_token_id=m["eos_token_id"],
        pad_token_id=m["pad_token_id"],
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    return model, cfg

def load_tokenizer(tokenizer_dir: str):
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer.json not found: {path!r} (run `make tok` first)")
    return PreTrainedTokenizerFast(tokenizer_file=path, **SPECIAL_TOKENS)
