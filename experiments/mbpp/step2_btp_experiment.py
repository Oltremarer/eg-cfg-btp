#!/usr/bin/env python3
"""
MBPP数据集的BTP实验 (Beam Search + Testing + Prioritized Experience Replay)

BTP算法包含两个主要阶段：
1. 阶段1: Beam Search采样 + 测试验证
2. 阶段2: 优先经验回放 (PPER) 训练

支持的功能：
- 本地模型的BTP实验
- 本地模型的BTP微调实验  
- OpenAI API的BTP实验
- DeepSeek API的BTP实验

使用示例：
1. 本地模型BTP实验：
   python experiments/mbpp/step2_btp_experiment.py --model deepseek-ai/deepseek-coder-1.3b-instruct --mode local

2. 本地模型微调：
   python experiments/mbpp/step2_btp_experiment.py --model deepseek-ai/deepseek-coder-1.3b-instruct --target-model deepseek-ai/deepseek-coder-1.3b-instruct --mode finetune

3. OpenAI实验：
   python experiments/mbpp/step2_btp_experiment.py --model gpt-4 --mode openai --api-key YOUR_KEY
"""

import os
import sys
import json
import argparse
import numpy as np
import random
import torch
import math
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# 导入共享基础类
from experiments.shared.base_experiment import Step2BTPExperiment  
from experiments.shared.dataset_configs import MBPP_CONFIG
from experiments.shared.common_utils import safe_execute_code, load_mbpp_problems

# 条件导入
try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        Trainer, 
        TrainingArguments,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    HF_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少HuggingFace依赖: {e}")
    HF_AVAILABLE = False

try:
    from eg_cfg.openai_utils import OpenAIClient, OpenAIInferenceError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# 项目相关导入
from eg_cfg.mbpp_utils import run_tests
if HF_AVAILABLE:
    from eg_cfg.model_utils import setup_device, load_model, load_tokenizer

from experiments.prompt_templates import get_model_prompt, detect_model_info, validate_model_compatibility
from experiments.shared.model_configs import get_model_config, get_optimal_generation_params


class ModelAdapter:
    """统一模型适配器 - 支持本地和API模型"""
    
    def __init__(self, model_name: str, model_type: str = "local", 
                 api_key: str = None, api_base: str = None, **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.api_base = api_base
        self.kwargs = kwargs
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self._setup_model()
    
    def _setup_model(self):
        """设置模型"""
        if self.model_type == "local" or self.model_type == "finetune":
            self._setup_local_model()
        elif self.model_type == "openai":
            self._setup_openai_model()
        elif self.model_type in ["deepseek", "api"]:
            self._setup_api_model()
    
    def _setup_local_model(self):
        """设置本地模型"""
        if not HF_AVAILABLE:
            raise ImportError("本地模型需要安装transformers库")
        
        print(f"🔧 加载本地模型: {self.model_name}")
        self.device = setup_device()
        self.model, self.tokenizer = load_model(self.model_name, self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _setup_openai_model(self):
        """设置OpenAI模型"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI模型需要安装openai相关依赖")
        
        print(f"🔧 配置OpenAI模型: {self.model_name}")
        self.client = OpenAIClient(api_key=self.api_key, model=self.model_name)
    
    def _setup_api_model(self):
        """设置API模型"""
        if not REQUESTS_AVAILABLE:
            raise ImportError("API模型需要安装requests库")
        
        print(f"🔧 配置API模型: {self.model_name}")
        self.api_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt: str, **generation_kwargs) -> List[Dict]:
        """统一生成接口"""
        if self.model_type == "local" or self.model_type == "finetune":
            return self._generate_local(prompt, **generation_kwargs)
        elif self.model_type == "openai":
            return self._generate_openai(prompt, **generation_kwargs)
        elif self.model_type in ["deepseek", "api"]:
            return self._generate_api(prompt, **generation_kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _generate_local(self, prompt: str, num_beams: int = 5, 
                       temperature: float = 0.8, max_tokens: int = 512,
                       **kwargs) -> List[Dict]:
        """本地模型生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        results = []
        sequences = outputs.sequences
        scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else None
        
        for i, sequence in enumerate(sequences):
            generated_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            code = generated_text[len(prompt):].strip()
            
            if scores is not None:
                log_prob = scores[i].item()
                possibility = min(math.exp(log_prob / len(sequence)), 1.0)
            else:
                log_prob = -10.0
                possibility = 0.5
            
            results.append({
                'code': code,
                'possibility': possibility,
                'log_prob': log_prob,
                'beam_rank': i,
                'sequence_length': len(sequence) - inputs['input_ids'].shape[1]
            })
        
        return results
    
    def _generate_openai(self, prompt: str, num_beams: int = 5, 
                        temperature: float = 0.8, **kwargs) -> List[Dict]:
        """OpenAI模型生成"""
        results = []
        
        for i in range(num_beams):
            try:
                response = self.client.generate_completion(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=512
                )
                
                results.append({
                    'code': response.get('content', ''),
                    'possibility': 0.8,  # OpenAI不提供具体概率
                    'log_prob': -5.0,
                    'beam_rank': i,
                    'sequence_length': len(response.get('content', ''))
                })
                
            except Exception as e:
                print(f"OpenAI生成失败 (beam {i}): {e}")
                results.append({
                    'code': '',
                    'possibility': 0.0,
                    'log_prob': -100.0,
                    'beam_rank': i,
                    'sequence_length': 0,
                    'error': str(e)
                })
        
        return results
    
    def _generate_api(self, prompt: str, num_beams: int = 5, 
                     temperature: float = 0.8, **kwargs) -> List[Dict]:
        """API模型生成"""
        results = []
        
        for i in range(num_beams):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 512
                }
                
                response = requests.post(
                    self.api_base or "https://api.deepseek.com/v1/chat/completions",
                    headers=self.api_headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    
                    results.append({
                        'code': content,
                        'possibility': 0.7,
                        'log_prob': -8.0,
                        'beam_rank': i,
                        'sequence_length': len(content)
                    })
                else:
                    raise Exception(f"API错误: {response.status_code}")
                    
            except Exception as e:
                print(f"API生成失败 (beam {i}): {e}")
                results.append({
                    'code': '',
                    'possibility': 0.0,
                    'log_prob': -100.0,
                    'beam_rank': i,
                    'sequence_length': 0,
                    'error': str(e)
                })
        
        return results


class P2ValueCalculator:
    """P2Value计算器 - 结合可能性和通过率"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def calculate_p2value(self, possibility: float, pass_rate: float) -> float:
        """计算P2Value = α * possibility + (1-α) * pass_rate"""
        return self.alpha * possibility + (1 - self.alpha) * pass_rate
    
    def calculate_p2value_extended(self, log_prob=None, sequence_length=None, 
                                 possibility=None, passed_tests=0, total_tests=1):
        """扩展P2Value计算，考虑更多因素"""
        if possibility is None and log_prob is not None:
            possibility = min(math.exp(log_prob / max(sequence_length, 1)), 1.0)
        
        pass_rate = passed_tests / max(total_tests, 1)
        
        if possibility is None:
            possibility = 0.5
        
        return self.calculate_p2value(possibility, pass_rate)


class PrioritizedSampler:
    """优先采样器 - 基于P2Value进行采样"""
    
    def __init__(self, sampling_method: str = "power", alpha: float = 1.0):
        self.sampling_method = sampling_method
        self.alpha = alpha
        
        if sampling_method not in ["power", "rank"]:
            raise ValueError(f"不支持的采样方法: {sampling_method}")
    
    def sample(self, experiences: List[Dict], batch_size: int) -> List[Dict]:
        """采样经验"""
        if len(experiences) <= batch_size:
            return experiences
        
        if self.sampling_method == "power":
            return self._power_sampling(experiences, batch_size)
        elif self.sampling_method == "rank":
            return self._rank_sampling(experiences, batch_size)
    
    def _power_sampling(self, experiences: List[Dict], batch_size: int) -> List[Dict]:
        """幂采样"""
        # 计算权重
        weights = []
        for exp in experiences:
            p2value = exp.get('p2value', 0.0)
            weight = max(p2value ** self.alpha, 1e-8)
            weights.append(weight)
        
        weights = np.array(weights)
        probabilities = weights / weights.sum()
        
        # 采样索引
        indices = np.random.choice(
            len(experiences), 
            size=batch_size, 
            replace=False, 
            p=probabilities
        )
        
        return [experiences[i] for i in indices]
    
    def _rank_sampling(self, experiences: List[Dict], batch_size: int) -> List[Dict]:
        """排序采样"""
        # 按P2Value排序
        sorted_experiences = sorted(
            experiences, 
            key=lambda x: x.get('p2value', 0.0), 
            reverse=True
        )
        
        # 计算排序权重
        weights = [1.0 / (rank + 1) ** self.alpha for rank in range(len(sorted_experiences))]
        weights = np.array(weights)
        probabilities = weights / weights.sum()
        
        # 采样
        indices = np.random.choice(
            len(sorted_experiences), 
            size=batch_size, 
            replace=False, 
            p=probabilities
        )
        
        return [sorted_experiences[i] for i in indices]


class ExperienceReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.p2calculator = P2ValueCalculator()
    
    def add_experience(self, experience: Dict):
        """添加经验"""
        # 计算P2Value
        experience['p2value'] = self.p2calculator.calculate_p2value_extended(
            possibility=experience.get('possibility', 0.5),
            passed_tests=experience.get('passed_tests', 0),
            total_tests=experience.get('total_tests', 1)
        )
        
        self.experiences.append(experience)
    
    def get_all_experiences(self) -> List[Dict]:
        """获取所有经验"""
        return list(self.experiences)
    
    def get_stats(self, include_samples: bool = False, max_samples: int = 10) -> Dict:
        """获取统计信息"""
        if not self.experiences:
            return {}
        
        p2values = [exp.get('p2value', 0.0) for exp in self.experiences]
        pass_rates = [exp.get('pass_rate', 0.0) for exp in self.experiences]
        
        stats = {
            'total_experiences': len(self.experiences),
            'avg_p2value': np.mean(p2values),
            'max_p2value': np.max(p2values), 
            'min_p2value': np.min(p2values),
            'avg_pass_rate': np.mean(pass_rates),
            'fully_passed_count': sum(1 for pr in pass_rates if pr >= 1.0),
            'zero_passed_count': sum(1 for pr in pass_rates if pr == 0.0)
        }
        
        # 如果需要包含样本数据，添加一些代表性样本
        if include_samples:
            samples = []
            
            # 获取通过率最高的样本
            best_experiences = sorted(self.experiences, key=lambda x: x.get('pass_rate', 0), reverse=True)[:max_samples//2]
            samples.extend(best_experiences)
            
            # 添加一些随机样本
            import random
            remaining_samples = max_samples - len(samples)
            if remaining_samples > 0 and len(self.experiences) > len(samples):
                random_experiences = random.sample(
                    [exp for exp in self.experiences if exp not in samples], 
                    min(remaining_samples, len(self.experiences) - len(samples))
                )
                samples.extend(random_experiences)
            
            stats['sample_experiences'] = samples
        
        return stats


class MBTPFineTuningManager:
    """MBPP BTP微调管理器"""
    
    def __init__(self, model_adapter: ModelAdapter, use_lora: bool = True, 
                 lora_config: Optional[Dict] = None, output_dir: str = "./mbpp_btp_checkpoints"):
        self.model_adapter = model_adapter
        self.use_lora = use_lora
        self.output_dir = output_dir
        self.lora_config = lora_config or {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1
        }
        
        if self.use_lora and HF_AVAILABLE:
            self._setup_lora()
    
    def _setup_lora(self):
        """设置LoRA微调"""
        if self.model_adapter.model_type not in ["local", "finetune"]:
            print("⚠️  LoRA微调仅支持本地模型")
            return
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            lora_dropout=self.lora_config['lora_dropout'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        self.model_adapter.model = get_peft_model(self.model_adapter.model, lora_config)
        print("✅ LoRA配置完成")
    
    def finetune_on_experiences(self, experiences: List[Dict], 
                               training_args: Optional[TrainingArguments] = None) -> None:
        """基于经验进行微调"""
        if self.model_adapter.model_type not in ["local", "finetune"]:
            print("⚠️  微调仅支持本地模型")
            return
        
        # 准备训练数据
        train_dataset = self._prepare_training_dataset(experiences)
        
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=self.output_dir,  # 使用自定义输出目录
                num_train_epochs=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                learning_rate=1e-6,  # 大幅减少学习率，从1e-4改为1e-6
                fp16=True,
                logging_steps=5,
                save_steps=100,
                remove_unused_columns=False,
            )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_adapter.tokenizer,
            mlm=False,
        )
        
        # 训练器
        trainer = Trainer(
            model=self.model_adapter.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        print(f"🚀 开始微调... 模型将保存到: {self.output_dir}")
        trainer.train()
        trainer.save_model()
        print("✅ 微调完成")
    
    def _prepare_training_dataset(self, experiences: List[Dict]) -> Dataset:
        """准备训练数据集"""
        texts = []
        
        for exp in experiences:
            instruction = f"Solve this programming problem:\n{exp['problem_text']}"
            response = exp['code']
            
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{self.model_adapter.tokenizer.eos_token}"
            texts.append(text)
        
        def tokenize_function(examples):
            if isinstance(examples['text'], str):
                examples['text'] = [examples['text']]
            
            tokenized = self.model_adapter.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors="pt"
            )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        
        return tokenized_dataset


class MBBPBTPExperiment(Step2BTPExperiment):
    """MBPP数据集的BTP实验 - 使用智能Prompt适配系统"""
    
    def __init__(self, model_name: str = None, model_type: str = "local", 
                 api_key: str = None, api_base: str = None,
                 sampling_method: str = "power", sampling_alpha: float = 1.0, 
                 p2value_alpha: float = 0.5, output_dir: str = "./mbpp_btp_checkpoints"):
        
        # 设置基本模型信息
        self.model_name = model_name or "deepseek-ai/deepseek-coder-1.3b-instruct"
        self.model_type = model_type
        self.api_key = api_key
        self.api_base = api_base
        self.output_dir = output_dir
        
        # BTP特定参数  
        self.sampling_method = sampling_method
        self.sampling_alpha = sampling_alpha
        self.p2value_alpha = p2value_alpha
        
        # 调用父类构造函数
        super().__init__(dataset_name="mbpp", model_name=self.model_name)
        
        # 初始化智能配置（在父类构造函数之后）
        self.model_info = detect_model_info(self.model_name)
        self.model_config = get_model_config(self.model_name)
        self.optimal_params = get_optimal_generation_params(self.model_name, "mbpp")
        
        # 验证模型兼容性
        compatibility = validate_model_compatibility(self.model_name, "mbpp")
        if compatibility["warnings"]:
            print("⚠️  模型兼容性警告:")
            for warning in compatibility["warnings"]:
                print(f"   - {warning}")
        
        if compatibility["recommendations"]:
            print("💡 优化建议:")
            for rec in compatibility["recommendations"]:
                print(f"   - {rec}")
        
        # 设置adapter
        self.adapter = ModelAdapter(
            model_name=self.model_name,
            model_type=self.model_type,
            api_key=self.api_key or "",  # 确保不是None
            api_base=self.api_base or "",  # 确保不是None
            **self.optimal_params  # 使用优化参数
        )
        
        # 初始化BTP组件
        self.experience_buffer = ExperienceReplayBuffer()
        self.sampler = PrioritizedSampler(sampling_method, sampling_alpha)
        self.p2calculator = P2ValueCalculator(p2value_alpha)
        
        # 微调管理器（如果需要）
        if model_type == "finetune":
            self.finetuning_manager = MBTPFineTuningManager(
                self.adapter, 
                use_lora=True, 
                output_dir=self.output_dir
            )
        else:
            self.finetuning_manager = None
        
        print(f"🚀 初始化完成:")
        print(f"   模型: {self.model_name}")
        print(f"   家族: {self.model_info.family.value}")
        print(f"   类型: {self.model_info.type.value}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   优化参数: {self.optimal_params}")
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载MBPP配置"""
        return MBPP_CONFIG
    
    def load_dataset(self) -> Dict[str, Any]:
        """加载MBPP数据集"""
        return load_mbpp_problems()
    
    def format_prompt(self, problem: Dict[str, Any]) -> str:
        """使用智能Prompt模板格式化问题"""
        
        # 临时禁用few-shot examples以修复0%通过率问题
        # TODO: 后续优化few-shot examples的prompt模板
        use_examples = False  # 原来是：self.model_config.use_examples
        examples = None
        
        if use_examples:
            # 为DeepSeek等模型准备few-shot examples
            examples = self._get_few_shot_examples()
        
        # 使用智能prompt引擎生成prompt
        prompt = get_model_prompt(
            model_name=self.model_name,
            dataset="mbpp", 
            problem=problem,
            system_prompt=None,  # 使用默认系统prompt
            use_examples=use_examples,
            examples=examples
        )
        
        # 如果返回的是messages格式（OpenAI/Claude），转换为字符串
        if isinstance(prompt, list):
            # 提取用户消息内容
            for msg in prompt:
                if msg["role"] == "user":
                    return msg["content"]
            return str(prompt)
        
        return prompt
    
    def _get_few_shot_examples(self) -> List[Dict[str, Any]]:
        """获取few-shot示例（特别针对DeepSeek等模型）"""
        
        # MBPP的经典示例，已经验证过效果
        examples = [
            {
                "problem": "Write a function to find the similar elements from the given two tuple lists.",
                "test_cases": [
                    "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                    "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                    "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
                ],
                "solution": """def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)"""
            },
            {
                "problem": "Write a python function to identify non-prime numbers.",
                "test_cases": [
                    "assert is_not_prime(2) == False",
                    "assert is_not_prime(10) == True", 
                    "assert is_not_prime(35) == True"
                ],
                "solution": """import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result"""
            },
            {
                "problem": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
                "test_cases": [
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
                    "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"
                ],
                "solution": """import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums"""
            }
        ]
        
        # 根据模型配置选择示例数量
        max_examples = self.model_config.preferred_examples_count
        return examples[:max_examples]
    
    def phase1_beam_search_sampling(self, problems_list: List[tuple], num_beams: int):
        """阶段1: Beam Search采样"""
        print("🔍 阶段1: Beam Search采样")
        
        for task_id, problem in tqdm(problems_list, desc="Beam Search采样"):
            prompt = self.format_prompt(problem)
            
            try:
                # 生成候选解
                candidates = self.adapter.generate(
                    prompt, 
                    num_beams=num_beams,
                    temperature=0.8,
                    max_tokens=512
                )
                
                # 测试每个候选解
                for candidate in candidates:
                    code = candidate['code']
                    if not code.strip():
                        continue
                    
                    try:
                        # 运行测试
                        test_results = run_tests(code, problem['test_list'])
                        passed_tests = sum(1 for r in test_results.values() if r.get('result', False))
                        total_tests = len(test_results)
                        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
                        
                        experience = {
                            'problem_id': str(task_id),
                            'problem_text': problem['text'],
                            'code': code,
                            'possibility': candidate['possibility'],
                            'pass_rate': pass_rate,
                            'passed_tests': passed_tests,
                            'total_tests': total_tests,
                            'test_results': test_results,
                            'beam_rank': candidate['beam_rank']
                        }
                        
                        self.experience_buffer.add_experience(experience)
                        
                    except Exception as e:
                        # 测试失败也要记录
                        experience = {
                            'problem_id': str(task_id),
                            'problem_text': problem['text'],
                            'code': code,
                            'possibility': candidate['possibility'],
                            'pass_rate': 0.0,
                            'passed_tests': 0,
                            'total_tests': len(problem.get('test_list', [])),
                            'error': str(e),
                            'beam_rank': candidate['beam_rank']
                        }
                        self.experience_buffer.add_experience(experience)
                        
            except Exception as e:
                print(f"⚠️  问题 {task_id} 生成失败: {e}")
                continue
    
    def phase2_pper_training(self, n_iterations: int, batch_size: int):
        """阶段2: 优先经验回放训练"""
        print(f"🎯 阶段2: 优先经验回放训练 ({n_iterations} 轮迭代)")
        
        if self.finetuning_manager is None:
            print("⚠️  跳过微调阶段（当前模式不支持微调）")
            return
        
        for iteration in range(n_iterations):
            print(f"\n📈 迭代 {iteration + 1}/{n_iterations}")
            
            # 获取所有经验
            all_experiences = self.experience_buffer.get_all_experiences()
            if len(all_experiences) == 0:
                print("⚠️  没有可用经验，跳过此轮迭代")
                continue
            
            # 优先采样
            sampled_experiences = self.sampler.sample(all_experiences, batch_size)
            print(f"📊 采样了 {len(sampled_experiences)} 个经验用于训练")
            
            # 执行微调
            try:
                self.finetuning_manager.finetune_on_experiences(sampled_experiences)
                print(f"✅ 迭代 {iteration + 1} 微调完成")
            except Exception as e:
                print(f"❌ 迭代 {iteration + 1} 微调失败: {e}")
                continue
    
    def get_experiment_results(self) -> Dict[str, Any]:
        """获取实验结果"""
        stats = self.experience_buffer.get_stats(include_samples=True, max_samples=20)
        all_experiences = self.experience_buffer.get_all_experiences()
        
        results = {
            'experiment_type': 'MBPP_BTP',
            'model_name': self.model_name,
            'mode': self.model_type,
            'target_model': self.model_name, # 因为微调模式下目标模型就是当前模型
            'sampling_method': self.sampling_method,
            'sampling_alpha': self.sampling_alpha,
            'p2value_alpha': self.p2value_alpha,
            'experience_stats': stats,
            'all_experiences': all_experiences,  # 保存所有生成的代码和结果
            'config': self.get_experiment_config()
        }
        
        # 打印一些生成的代码样本用于调试
        print("\n" + "="*80)
        print("🔍 生成代码样本分析 (用于调试0%通过率问题)")
        print("="*80)
        
        if 'sample_experiences' in stats and stats['sample_experiences']:
            samples = stats['sample_experiences'][:5]  # 只看前5个
            for i, exp in enumerate(samples):
                print(f"\n📝 样本 {i+1}:")
                print(f"   问题ID: {exp.get('problem_id', 'N/A')}")
                print(f"   通过率: {exp.get('pass_rate', 0):.2f}")
                print(f"   生成概率: {exp.get('possibility', 0):.4f}")
                print(f"   生成代码:")
                print("   " + "-"*60)
                code_lines = str(exp.get('code', '')).split('\n')
                for line in code_lines[:10]:  # 只显示前10行
                    print(f"   {line}")
                if len(code_lines) > 10:
                    print(f"   ... (还有 {len(code_lines)-10} 行)")
                print("   " + "-"*60)
                
                # 显示测试结果
                if 'test_results' in exp and exp['test_results']:
                    test_results = exp['test_results']
                    passed = sum(1 for r in test_results.values() if r.get('result', False))
                    total = len(test_results)
                    print(f"   测试结果: {passed}/{total} 通过")
                    
                    # 显示失败的测试（如果有）
                    failed_tests = [k for k, v in test_results.items() if not v.get('result', False)]
                    if failed_tests:
                        print(f"   失败测试: {failed_tests[:3]}")  # 只显示前3个失败测试
        
        print("="*80)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='MBPP数据集的BTP实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 本地模型BTP实验:
   python experiments/mbpp/step2_btp_experiment.py \\
     --model deepseek-ai/deepseek-coder-1.3b-instruct \\
     --mode local --max-problems 50

2. 本地模型微调（使用默认输出目录）:
   python experiments/mbpp/step2_btp_experiment.py \\
     --model deepseek-ai/deepseek-coder-1.3b-instruct \\
     --mode finetune --max-problems 100

3. 本地模型微调（自定义输出目录）:
   python experiments/mbpp/step2_btp_experiment.py \\
     --model deepseek-ai/deepseek-coder-1.3b-instruct \\
     --mode finetune --max-problems 100 \\
     --output-dir ./my_custom_checkpoints

4. OpenAI实验:
   python experiments/mbpp/step2_btp_experiment.py \\
     --model gpt-4 --mode openai \\
     --api-key YOUR_KEY --max-problems 30

5. DeepSeek API实验:
   python experiments/mbpp/step2_btp_experiment.py \\
     --model deepseek-chat --mode deepseek \\
     --api-key YOUR_KEY --max-problems 30
        """)
    
    # 基本参数
    parser.add_argument('--model', type=str, required=True,
                       help='模型名称或路径')
    parser.add_argument('--mode', type=str, default='local',
                       choices=['local', 'finetune', 'openai', 'deepseek'],
                       help='实验模式')
    parser.add_argument('--target-model', type=str, default=None,
                       help='目标微调模型（仅微调模式需要）')
    
    # API参数
    parser.add_argument('--api-key', type=str,
                       help='API密钥')
    
    # 实验参数
    parser.add_argument('--max-problems', type=int, default=50,
                       help='最大问题数量')
    parser.add_argument('--num-beams', type=int, default=5,
                       help='Beam Search数量')
    parser.add_argument('--n-iterations', type=int, default=2,
                       help='PPER训练迭代次数')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='训练批大小')
    parser.add_argument('--output-dir', type=str, default='./mbpp_btp_checkpoints',
                       help='模型保存目录（仅微调模式）')
    
    # BTP算法参数
    parser.add_argument('--sampling-method', type=str, default='power',
                       choices=['power', 'rank'],
                       help='采样方法')
    parser.add_argument('--sampling-alpha', type=float, default=1.0,
                       help='采样α参数')
    parser.add_argument('--p2value-alpha', type=float, default=0.5,
                       help='P2Value权重α')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试日志')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 打印配置
    print("🚀 MBPP BTP实验配置:")
    print(f"  模型: {args.model}")
    print(f"  模式: {args.mode}")
    print(f"  最大问题数: {args.max_problems}")
    print(f"  采样方法: {args.sampling_method}")
    print(f"  采样Alpha: {args.sampling_alpha}")
    print(f"  P2Value Alpha: {args.p2value_alpha}")
    if args.mode == "finetune":
        print(f"  输出目录: {args.output_dir}")
    
    # 创建实验实例
    experiment = MBBPBTPExperiment(
        model_name=args.model,
        model_type=args.mode,  # 修复参数名：mode -> model_type
        api_key=args.api_key,
        api_base=None,  # API base 参数在 ModelAdapter 中处理
        sampling_method=args.sampling_method,
        sampling_alpha=args.sampling_alpha,
        p2value_alpha=args.p2value_alpha,
        output_dir=args.output_dir # 传递output_dir参数
    )
    
    # 运行实验
    try:
        results = experiment.run_experiment(
            max_problems=args.max_problems,
            num_beams=args.num_beams,
            n_iterations=args.n_iterations,
            batch_size=args.batch_size
        )
        
        # 保存结果
        result_file = experiment.save_results(results, "btp_experiment")
        
        print("\n✅ 实验完成!")
        print(f"📊 实验统计:")
        for key, value in results['experience_stats'].items():
            print(f"  {key}: {value}")
        
        print(f"📁 结果已保存到: {result_file}")
        return 0
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 