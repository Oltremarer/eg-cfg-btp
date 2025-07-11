import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .consts import *
import math


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    return device


# In the model: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
# https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/discussions/8
# $HOME/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11/modeling_deepseek.py
# line 1728:
# -- max_cache_length = past_key_values.get_max_length()
# ++ max_cache_length = past_key_values.get_max_cache_shape()
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)


def load_model(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == DEEPSEEK_CODER_V2_LITE_INSTRUCT_MODEL_NAME:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(
            device
        )
    #model = torch.compile(model)
    return model, tokenizer


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


def extract_new_tokens(tokenizer, input_ids: torch.Tensor, prompt_input_ids_len) -> str:
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("Expected input_ids to have shape (1, sequence_length)")

    new_token_ids = input_ids[:, prompt_input_ids_len:]
    new_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]
    return new_text, new_token_ids


def calculate_tokens_length(tokenizer, prompt):
    prompt_token_ids = tokenizer(prompt, return_tensors="pt")
    prompt_input_ids = prompt_token_ids["input_ids"]  # shape: (1, prompt_len)
    prompt_input_ids_len = prompt_input_ids.shape[1]
    return prompt_input_ids_len


def convert_logprobs_dist_dict_to_tokenizer_prob_dist(tokenizer, logprob_dist_dict):
    prob_dist_dict = {
        token_idx: math.exp(logprob) for token_idx, logprob in logprob_dist_dict.items()
    }
    vocab_size = tokenizer.vocab_size
    prob_dist = torch.zeros(vocab_size)

    for token_idx, prob in prob_dist_dict.items():
        prob_dist[token_idx] = prob

    return prob_dist
