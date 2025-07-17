import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .consts import *
import math
import os
import json


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    return device


def load_model(model_name: str, device):
    # 检查是否是本地路径
    if os.path.exists(model_name) and os.path.isdir(model_name):
        # 本地检查点，需要特殊处理
        print(f"🔧 加载本地检查点: {model_name}")
        
        # 优先查找adapter_config.json（LoRA adapter），否则查找config.json
        config_path = os.path.join(model_name, "adapter_config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(model_name, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 检查是否是LoRA检查点
                if "peft_type" in config or "base_model_name_or_path" in config:
                    print("🔧 检测到LoRA微调检查点，使用PEFT加载")
                    try:
                        from peft import PeftModel
                        
                        # 加载tokenizer
                        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        
                        # 获取基础模型路径
                        base_model_path = config.get("base_model_name_or_path", "deepseek-ai/deepseek-coder-1.3b-instruct")
                        print(f"🔧 加载基础模型: {base_model_path}")
                        
                        # 加载基础模型
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_path,
                            trust_remote_code=True,
                            torch_dtype="auto"
                        ).to(device)
                        
                        # 加载LoRA适配器
                        model = PeftModel.from_pretrained(base_model, model_name)
                        print("✅ LoRA适配器加载成功")
                        
                        return model, tokenizer
                        
                    except ImportError:
                        print("⚠️  PEFT库未安装，尝试直接加载")
                    except Exception as e:
                        print(f"⚠️  LoRA加载失败: {e}，尝试直接加载")
            except Exception as e:
                print(f"⚠️  读取配置文件失败: {e}，尝试直接加载")
        
        # 如果不是LoRA检查点或加载失败，尝试直接加载
        print("🔧 尝试直接加载检查点")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype="auto"
        ).to(device)
    else:
        # 远程模型
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


def extract_new_tokens(tokenizer, input_ids: torch.Tensor, prompt_input_ids_len):
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("Expected input_ids to have shape (1, sequence_length)")

    new_token_ids = input_ids[:, prompt_input_ids_len:]
    new_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
    return new_text, new_token_ids


def calculate_tokens_length(tokenizer, prompt):
    prompt_token_ids = tokenizer(prompt, return_tensors="pt")
    prompt_input_ids = prompt_token_ids["input_ids"]  # shape: (1pt_len)
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
