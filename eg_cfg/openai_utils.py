import os
import json
import re
import math
from typing import Optional, List, Dict, Any
import requests
from .consts import *

class OpenAIInferenceError(Exception):
    pass

def extract_python_code(text: str) -> Optional[str]:
    """从文本中提取Python代码块"""
    # 匹配```python 代码块
    python_pattern = r'```python\s*\n(.*?)```'
    match = re.search(python_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 匹配普通``` 代码块
    code_pattern = r'```\s*\n(.*?)```'
    match = re.search(code_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""

class OpenAIClient:
    """OpenAI API客户端，支持OpenAI和兼容的API服务"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_code(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.8,
        n: int = 1,
        stop: Optional[List[str]] = None
    ) -> List[str]:
        """生成代码补全"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # 提取生成的文本
            completions = []
            for choice in data["choices"]:
                content = choice["message"]["content"]
                
                # 尝试提取Python代码块
                extracted_code = extract_python_code(content)
                if extracted_code:
                    completions.append(extracted_code)
                else:
                    # 如果没有代码块，返回原始内容
                    completions.append(content.strip())
            
            return completions
            
        except requests.RequestException as e:
            raise OpenAIInferenceError(f"API request failed: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise OpenAIInferenceError(f"Invalid API response: {e}")
    
    def generate_code_with_probs(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.8,
        n: int = 5,
        logprobs: bool = True,
        top_logprobs: int = 5,
        stop: Optional[List[str]] = None
    ) -> List[Dict]:
        """生成代码并返回概率信息 - 新增方法"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for i, choice in enumerate(data["choices"]):
                content = choice["message"]["content"]
                extracted_code = extract_python_code(content)
                
                # 计算真实的序列概率
                if "logprobs" in choice and choice["logprobs"] and choice["logprobs"]["content"]:
                    logprobs_data = choice["logprobs"]["content"]
                    # 计算平均log概率
                    token_logprobs = [token["logprob"] for token in logprobs_data if token["logprob"] is not None]
                    if token_logprobs:
                        avg_log_prob = sum(token_logprobs) / len(token_logprobs)
                        possibility = min(math.exp(avg_log_prob), 1.0)
                    else:
                        avg_log_prob = -10.0
                        possibility = 0.1
                else:
                    token_logprobs = []
                    avg_log_prob = -10.0
                    possibility = 0.1  # 降低默认值，之前是0.8
                
                results.append({
                    'code': extracted_code or content.strip(),
                    'possibility': possibility,
                    'log_prob': avg_log_prob,
                    'beam_rank': i,
                    'sequence_length': len(content),
                    'token_logprobs': token_logprobs
                })
            
            return results
            
        except requests.RequestException as e:
            raise OpenAIInferenceError(f"API request failed: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise OpenAIInferenceError(f"Invalid API response: {e}")
    
    def generate_code_completion(
        self,
        prefix: str,
        suffix: str = "",
        max_tokens: int = 512,
        temperature: float = 0.8,
        n: int = 1
    ) -> List[str]:
        """使用代码补全模式生成代码"""
        
        # 对于聊天模型，我们构造一个适当的提示
        prompt = f"请完成以下Python代码：\n\n```python\n{prefix}"
        if suffix:
            prompt += f"\n# TODO: 在这里添加代码\n{suffix}"
        prompt += "\n```\n\n请只返回需要补全的代码部分。"
        
        return self.generate_code(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n
        )

# 为了兼容现有代码，创建一些包装函数
def openai_simple_query(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int = MAX_NEW_TOKENS,
    n: int = 1,
    api_key: str = None,
    base_url: str = None,
    **kwargs
) -> List[str]:
    """简单的OpenAI查询，兼容现有接口"""
    
    client = OpenAIClient(api_key=api_key, base_url=base_url, model=model_name)
    return client.generate_code(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )

def openai_beam_search_batch(
    prompt: str,
    model_name: str,
    temperature: float,
    candidates_count: int = 3,
    max_tokens: int = MAX_NEW_TOKENS,
    api_key: str = None,
    base_url: str = None,
    **kwargs
) -> List[str]:
    """OpenAI批量生成，模拟beam search"""
    
    client = OpenAIClient(api_key=api_key, base_url=base_url, model=model_name)
    return client.generate_code(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=candidates_count
    ) 