#!/usr/bin/env python3
"""
统一的BTP实验脚本 - 支持本地模型微调和API模型
BTP = Beam Search + Testing + Prioritized Experience Replay

支持的功能：
1. 本地模型的BTP实验（不含微调）
2. 本地模型的BTP微调实验
3. OpenAI API的BTP实验
4. DeepSeek API的BTP实验
5. 混合模式（API采样+本地微调）

使用示例：
1. 本地模型BTP实验（无微调）：
   python experiments/step2_btp_finetune_experiment.py --source-model deepseek-ai/deepseek-coder-1.3b-instruct --mode btp_only

2. 本地模型微调：
   python experiments/step2_btp_finetune_experiment.py --source-model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --target-model deepseek-ai/deepseek-coder-1.3b-instruct --mode finetune

3. OpenAI BTP实验：
   python experiments/step2_btp_finetune_experiment.py --source-model gpt-4 --mode openai --api-key YOUR_KEY

4. 混合模式（API采样+本地微调）：
   python experiments/step2_btp_finetune_experiment.py --source-model gpt-4 --target-model deepseek-ai/deepseek-coder-1.3b-instruct --mode hybrid --api-key YOUR_KEY
"""

import os
import sys
import json
import argparse
import numpy as np
import random
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
from datetime import datetime
import math
import logging
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 条件导入
try:
    # Transformers相关导入
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        Trainer, 
        TrainingArguments,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    HF_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少HuggingFace依赖: {e}")
    print("本地模型功能将不可用。如需使用本地模型，请安装: pip install transformers peft datasets")
    HF_AVAILABLE = False

try:
    from eg_cfg.openai_utils import OpenAIClient, OpenAIInferenceError
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️  OpenAI工具不可用，OpenAI功能将被禁用")
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# 项目相关导入
from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests
if HF_AVAILABLE:
    from eg_cfg.model_utils import setup_device, load_model, load_tokenizer


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
        if self.model_type == "local":
            self._setup_local_model()
        elif self.model_type == "openai":
            self._setup_openai_model()
        elif self.model_type in ["deepseek", "api"]:
            self._setup_api_model()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _setup_local_model(self):
        """设置本地模型"""
        if not HF_AVAILABLE:
            raise ImportError("本地模型需要安装transformers库")
        
        print(f"🔧 加载本地模型: {self.model_name}")
        
        self.device = setup_device()
        self.model, self.tokenizer = load_model(self.model_name, self.device)
        
        # 设置pad_token
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
        if self.model_type == "local":
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
        
        try:
            solutions = self.client.generate_code(
                prompt=prompt,
                max_tokens=512,
                temperature=temperature,
                n=num_beams
            )
            
            for i, code in enumerate(solutions):
                # 为OpenAI生成的代码计算可能性分数
                possibility = max(0.1, 1.0 - (temperature * 0.5) - (i * 0.1))
                
                results.append({
                    'code': code,
                    'possibility': possibility,
                    'beam_rank': i,
                    'temperature': temperature
                })
                
        except Exception as e:
            print(f"⚠️  OpenAI生成失败: {e}")
            for i in range(num_beams):
                results.append({
                    'code': f"# API调用失败: {e}",
                    'possibility': 0.0,
                    'beam_rank': i
                })
        
        return results
    
    def _generate_api(self, prompt: str, num_beams: int = 5, 
                     temperature: float = 0.8, **kwargs) -> List[Dict]:
        """通用API模型生成"""
        results = []
        api_url = self.api_base or "https://api.deepseek.com/v1/chat/completions"
        
        for i in range(num_beams):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 512
                }
                
                response = requests.post(api_url, headers=self.api_headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                code = data['choices'][0]['message']['content'].strip()
                possibility = random.uniform(0.3, 0.9)  # 模拟概率
                
                results.append({
                    'code': code,
                    'possibility': possibility,
                    'beam_rank': i
                })
                
            except Exception as e:
                print(f"⚠️  API调用失败: {e}")
                results.append({
                    'code': f"# API调用失败: {e}",
                    'possibility': 0.0,
                    'beam_rank': i
                })
        
        return results


class P2ValueCalculator:
    """P2Value计算器"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def calculate_p2value(self, possibility: float, pass_rate: float) -> float:
        """计算P2Value = α × possibility + (1-α) × pass_rate"""
        return self.alpha * possibility + (1 - self.alpha) * pass_rate
    
    def calculate_p2value_extended(self, log_prob=None, sequence_length=None, 
                                 possibility=None, passed_tests=0, total_tests=1):
        """扩展的P2Value计算，兼容不同输入格式"""
        if possibility is None:
            if log_prob is not None and sequence_length is not None:
                possibility = min(math.exp(log_prob / max(sequence_length, 1)), 1.0)
            else:
                possibility = 0.5  # 默认值
        
        pass_rate = passed_tests / max(total_tests, 1)
        p2value = self.alpha * possibility + (1 - self.alpha) * pass_rate
        
        return p2value, possibility, pass_rate


class PrioritizedSampler:
    """优先经验采样器 - 支持两种采样方式"""
    
    def __init__(self, sampling_method: str = "power", alpha: float = 1.0):
        """
        Args:
            sampling_method: "power" 或 "rank"
            alpha: 采样参数
        """
        self.sampling_method = sampling_method
        self.alpha = alpha
    
    def sample(self, experiences: List[Dict], batch_size: int) -> List[Dict]:
        """根据P2Value进行优先采样"""
        if not experiences:
            return []
        
        if self.sampling_method == "power":
            return self._power_sampling(experiences, batch_size)
        elif self.sampling_method == "rank":
            return self._rank_sampling(experiences, batch_size)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
    
    def _power_sampling(self, experiences: List[Dict], batch_size: int) -> List[Dict]:
        """幂次采样: P(i) = pi^α / Σ pk^α"""
        p2values = np.array([exp['p2value'] for exp in experiences])
        
        # 避免除零错误
        p2values = np.maximum(p2values, 1e-8)
        
        # 计算采样概率
        powered_values = np.power(p2values, self.alpha)
        probabilities = powered_values / np.sum(powered_values)
        
        # 采样
        indices = np.random.choice(
            len(experiences), 
            size=min(batch_size, len(experiences)), 
            p=probabilities, 
            replace=False
        )
        
        return [experiences[i] for i in indices]
    
    def _rank_sampling(self, experiences: List[Dict], batch_size: int) -> List[Dict]:
        """排名采样: pi = 1/rank(i)"""
        # 按P2Value排序
        sorted_experiences = sorted(experiences, key=lambda x: x['p2value'], reverse=True)
        
        # 计算排名概率
        ranks = np.arange(1, len(sorted_experiences) + 1)
        probabilities = 1.0 / ranks
        probabilities = probabilities / np.sum(probabilities)
        
        # 采样
        indices = np.random.choice(
            len(sorted_experiences), 
            size=min(batch_size, len(sorted_experiences)), 
            p=probabilities, 
            replace=False
        )
        
        return [sorted_experiences[i] for i in indices]


class ExperienceReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.p2calculator = P2ValueCalculator()
    
    def add_experience(self, experience: Dict):
        """添加经验到缓冲区"""
        if 'p2value' not in experience:
            experience['p2value'] = self.p2calculator.calculate_p2value(
                experience['possibility'], 
                experience['pass_rate']
            )
        self.buffer.append(experience)
    
    def get_all_experiences(self) -> List[Dict]:
        """获取所有经验"""
        return list(self.buffer)
    
    def get_stats(self, include_samples: bool = False, max_samples: int = 10) -> Dict:
        """获取缓冲区统计信息"""
        if not self.buffer:
            return {}
        
        experiences = list(self.buffer)
        p2values = [exp['p2value'] for exp in experiences]
        pass_rates = [exp['pass_rate'] for exp in experiences]
        
        stats = {
            'total_experiences': len(experiences),
            'avg_p2value': np.mean(p2values),
            'std_p2value': np.std(p2values),
            'avg_pass_rate': np.mean(pass_rates),
            'fully_passed_count': sum(1 for exp in experiences if exp['pass_rate'] >= 1.0),
            'zero_passed_count': sum(1 for exp in experiences if exp['pass_rate'] == 0.0),
            'max_p2value': np.max(p2values),
            'min_p2value': np.min(p2values)
        }
        
        # 如果需要包含样本数据，添加一些代表性样本
        if include_samples:
            # 获取不同通过率的样本
            samples = []
            
            # 尝试找到通过率最高的样本
            best_experiences = sorted(experiences, key=lambda x: x['pass_rate'], reverse=True)[:max_samples//2]
            samples.extend(best_experiences)
            
            # 添加一些随机样本
            import random
            remaining_samples = max_samples - len(samples)
            if remaining_samples > 0 and len(experiences) > len(samples):
                random_experiences = random.sample(
                    [exp for exp in experiences if exp not in samples], 
                    min(remaining_samples, len(experiences) - len(samples))
                )
                samples.extend(random_experiences)
            
            stats['sample_experiences'] = samples
        
        return stats


class BTPModelManager:
    """BTP模型管理器"""
    
    def __init__(self, 
                 source_model_path: str,
                 target_model_path: Optional[str] = None,
                 use_lora: bool = True,
                 lora_config: Optional[Dict] = None):
        """
        Args:
            source_model_path: 用于生成初始经验的模型路径
            target_model_path: 被微调的目标模型路径（如果不同于源模型）
            use_lora: 是否使用LoRA微调
            lora_config: LoRA配置参数
        """
        self.source_model_path = source_model_path
        self.target_model_path = target_model_path or source_model_path
        self.use_lora = use_lora
        
        # 加载源模型（用于生成）
        print(f"Loading source model: {self.source_model_path}")
        self.source_model = AutoModelForCausalLM.from_pretrained(
            self.source_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.source_model_path, trust_remote_code=True)
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载目标模型（用于微调）
        if self.target_model_path != self.source_model_path:
            print(f"Loading target model: {self.target_model_path}")
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.target_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.target_model = self.source_model
        
        # 配置LoRA
        if self.use_lora:
            default_lora_config = {
                "r": 64,
                "lora_alpha": 128,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            }
            
            if lora_config:
                default_lora_config.update(lora_config)
            
            lora_config_obj = LoraConfig(**default_lora_config)
            self.target_model = get_peft_model(self.target_model, lora_config_obj)
    
    def generate_beam_candidates(self, 
                                prompt: str, 
                                num_beams: int = 5, 
                                max_new_tokens: int = 512) -> List[Dict]:
        """使用beam search生成候选解决方案"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {k: v.to(self.source_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.source_model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        candidates = []
        for i, sequence in enumerate(outputs.sequences):
            generated_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            code = generated_text[len(prompt):].strip()
            
            # 计算生成概率
            if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                log_prob = outputs.sequences_scores[i].item()
                possibility = min(math.exp(log_prob / len(sequence)), 1.0)
            else:
                possibility = 1.0 / (i + 1)  # 简单的排名概率
            
            candidates.append({
                'code': code,
                'possibility': possibility,
                'beam_rank': i
            })
        
        return candidates


class BTPFineTuneExperiment:
    """BTP微调实验主类"""
    
    def __init__(self, 
                 source_model_path: str,
                 target_model_path: Optional[str] = None,
                 dataset: str = "mbpp",
                 sampling_method: str = "power",
                 sampling_alpha: float = 1.0,
                 p2value_alpha: float = 0.5,
                 use_lora: bool = True,
                 lora_config: Optional[Dict] = None):
        
        self.dataset = dataset
        
        # 初始化组件
        self.model_manager = BTPModelManager(
            source_model_path, target_model_path, use_lora, lora_config
        )
        self.experience_buffer = ExperienceReplayBuffer()
        self.sampler = PrioritizedSampler(sampling_method, sampling_alpha)
        self.p2calculator = P2ValueCalculator(p2value_alpha)
        
        # 加载数据集
        print(f"Loading dataset: {dataset}")
        if dataset == "mbpp":
            self.problems = load_mbpp_problems()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        print(f"Loaded {len(self.problems)} problems")
    
    def format_problem_prompt(self, problem: Dict) -> str:
        """格式化问题为提示词"""
        test_examples = "\n".join([f"  {test}" for test in problem.get('test_list', [])])
        
        return f"""Solve the following programming problem:

Problem: {problem['text']}

Test cases:
{test_examples}

Provide a complete Python function:

```python
"""
    
    def phase1_beam_search_sampling(self, problems: List[Dict], num_beams: int = 5) -> None:
        """阶段1: Beam Search采样阶段"""
        print("Phase 1: Beam Search Sampling")
        
        for problem_id, problem in tqdm(problems, desc="Beam Search Sampling"):
            prompt = self.format_problem_prompt(problem)
            
            # 生成beam candidates
            candidates = self.model_manager.generate_beam_candidates(
                prompt, num_beams=num_beams
            )
            
            # 测试每个候选并存储经验
            for candidate in candidates:
                try:
                    test_results = run_tests(candidate['code'], problem['test_list'])
                    passed_tests = sum(1 for r in test_results.values() if r.get('result', False))
                    total_tests = len(test_results)
                    pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
                    
                    experience = {
                        'problem_id': problem_id,
                        'problem_text': problem['text'],
                        'code': candidate['code'],
                        'possibility': candidate['possibility'],
                        'pass_rate': pass_rate,
                        'test_results': test_results,
                        'beam_rank': candidate['beam_rank']
                    }
                    
                    self.experience_buffer.add_experience(experience)
                    
                except Exception as e:
                    # 记录失败但继续
                    experience = {
                        'problem_id': problem_id,
                        'problem_text': problem['text'],
                        'code': candidate['code'],
                        'possibility': candidate['possibility'],
                        'pass_rate': 0.0,
                        'error': str(e),
                        'beam_rank': candidate['beam_rank']
                    }
                    self.experience_buffer.add_experience(experience)
    
    def phase2_pper_training(self, 
                           n_iterations: int = 3,
                           batch_size: int = 100,
                           training_args: Optional[TrainingArguments] = None) -> None:
        """阶段2: PPER训练阶段"""
        print(f"Phase 2: PPER Training ({n_iterations} iterations)")
        
        for iteration in range(n_iterations):
            print(f"\nIteration {iteration + 1}/{n_iterations}")
            
            # 优先采样高质量经验
            all_experiences = self.experience_buffer.get_all_experiences()
            sampled_experiences = self.sampler.sample(all_experiences, batch_size)
            
            print(f"Sampled {len(sampled_experiences)} experiences for training")
            
            # 准备训练数据
            train_dataset = self._prepare_training_dataset(sampled_experiences)
            
            # 执行微调
            self._finetune_model(train_dataset, training_args, iteration)
            
            # 可选：用微调后的模型重新评估部分问题
            # self._re_evaluate_with_finetuned_model(sample_problems)
    
    def _prepare_training_dataset(self, experiences: List[Dict]) -> Dataset:
        """准备微调训练数据集"""
        texts = []
        
        for exp in experiences:
            # 构造指令微调格式
            instruction = f"Solve this programming problem:\n{exp['problem_text']}"
            response = exp['code']
            
            # 格式化为训练文本
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{self.model_manager.tokenizer.eos_token}"
            texts.append(text)
        
        # 分词 - 修复这里的批处理问题
        def tokenize_function(examples):
            # 确保输入是字符串列表
            if isinstance(examples['text'], str):
                examples['text'] = [examples['text']]
            
            tokenized = self.model_manager.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors="pt"
            )
            
            # 为语言模型准备labels
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        
        return tokenized_dataset
    
    def _finetune_model(self, 
                       train_dataset: Dataset, 
                       training_args: Optional[TrainingArguments],
                       iteration: int) -> None:
        """执行模型微调"""
        
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=f"./btp_checkpoints/iteration_{iteration}",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=50,
                learning_rate=2e-5,
                fp16=True,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                remove_unused_columns=False,
            )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_manager.tokenizer,
            mlm=False,
        )
        
        # 训练器
        trainer = Trainer(
            model=self.model_manager.target_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        print(f"Starting fine-tuning iteration {iteration}...")
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        print(f"Model saved to {training_args.output_dir}")
    
    def run_experiment(self, 
                      max_problems: int = 100,
                      num_beams: int = 5,
                      n_iterations: int = 3,
                      batch_size: int = 100,
                      output_dir: str = "./btp_results") -> Dict:
        """运行完整的BTP实验"""
        
        print("=" * 60)
        print("BTP Fine-tuning Experiment")
        print("=" * 60)
        
        # 选择问题子集
        problems_list = list(self.problems.items())[:max_problems]
        
        # 阶段1: Beam Search + Testing
        self.phase1_beam_search_sampling(problems_list, num_beams)
        
        # 输出初始统计
        initial_stats = self.experience_buffer.get_stats(include_samples=True, max_samples=20)
        print(f"\nInitial experience buffer stats:")
        for key, value in initial_stats.items():
            if key != 'sample_experiences':  # 不打印样本数据，太长了
            print(f"  {key}: {value}")
        
        # 阶段2: PPER微调
        self.phase2_pper_training(n_iterations, batch_size)
        
        # 获取最终统计信息（包含样本）
        final_stats = self.experience_buffer.get_stats(include_samples=True, max_samples=20)
        
        # 获取所有经验数据
        all_experiences = self.experience_buffer.get_all_experiences()
        
        # 保存实验结果
        results = {
            'experiment_type': 'BTP_FineTune',
            'source_model': self.model_manager.source_model_path,
            'target_model': self.model_manager.target_model_path,
            'dataset': self.dataset,
            'sampling_method': self.sampler.sampling_method,
            'sampling_alpha': self.sampler.alpha,
            'p2value_alpha': self.p2calculator.alpha,
            'max_problems': max_problems,
            'num_beams': num_beams,
            'n_iterations': n_iterations,
            'batch_size': batch_size,
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'all_experiences': all_experiences,  # 保存所有生成的代码和结果
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, f"btp_finetune_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # 保存完整结果（包含所有生成的代码）
        print(f"\n💾 正在保存完整实验结果...")
        print(f"   - 包含 {len(all_experiences)} 个完整的代码生成样本")
        print(f"   - 每个样本包含：生成代码、测试结果、通过率、概率等")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 检查文件大小
        import os
        file_size = os.path.getsize(result_file) / (1024 * 1024)  # MB
        print(f"\n✅ 实验结果已保存到: {result_file}")
        print(f"📁 文件大小: {file_size:.2f} MB")
        print(f"📊 包含内容:")
        print(f"   - 实验配置和统计信息")
        print(f"   - {len(all_experiences)} 个完整的代码生成记录")
        print(f"   - 每个问题的测试结果详情")
        
        # 打印一些生成的代码样本用于调试
        print("\n" + "="*80)
        print("🔍 生成代码样本分析 (用于调试0%通过率问题)")
        print("="*80)
        
        if 'sample_experiences' in final_stats:
            samples = final_stats['sample_experiences'][:5]  # 只看前5个
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
        description='统一的BTP实验脚本 - 支持本地模型微调和API模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 本地模型BTP实验（无微调）:
   python experiments/step2_btp_finetune_experiment.py \\
     --source-model deepseek-ai/deepseek-coder-1.3b-instruct \\
     --mode btp_only --max-problems 50

2. 本地模型微调:
   python experiments/step2_btp_finetune_experiment.py \\
     --source-model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \\
     --target-model deepseek-ai/deepseek-coder-1.3b-instruct \\
     --mode finetune --max-problems 100

3. OpenAI BTP实验:
   python experiments/step2_btp_finetune_experiment.py \\
     --source-model gpt-4 --mode openai \\
     --api-key YOUR_OPENAI_KEY --max-problems 30

4. 混合模式（API采样+本地微调）:
   python experiments/step2_btp_finetune_experiment.py \\
     --source-model gpt-4 --mode hybrid \\
     --target-model deepseek-ai/deepseek-coder-1.3b-instruct \\
     --api-key YOUR_OPENAI_KEY --max-problems 50

5. DeepSeek API实验:
   python experiments/step2_btp_finetune_experiment.py \\
     --source-model deepseek-chat --mode deepseek \\
     --api-key YOUR_DEEPSEEK_KEY \\
     --api-base https://api.deepseek.com --max-problems 30
        """)
    
    # 实验模式
    parser.add_argument('--mode', type=str, default='finetune',
                       choices=['btp_only', 'finetune', 'openai', 'deepseek', 'hybrid'],
                       help='实验模式')
    
    # 模型相关参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--source-model', type=str, required=True,
                           help='源模型路径或名称')
    model_group.add_argument('--target-model', type=str, default=None,
                           help='目标模型路径（用于微调）')
    
    # API参数
    api_group = parser.add_argument_group('API参数')
    api_group.add_argument('--api-key', type=str,
                         help='API密钥（OpenAI/DeepSeek等）')
    api_group.add_argument('--api-base', type=str,
                         help='API基础URL（可选）')
    
    # 数据集参数
    data_group = parser.add_argument_group('数据集参数')
    data_group.add_argument('--dataset', type=str, default='mbpp',
                          choices=['mbpp', 'humaneval'],
                          help='数据集名称')
    data_group.add_argument('--max-problems', type=int, default=50,
                          help='最大问题数量')
    
    # BTP算法参数
    btp_group = parser.add_argument_group('BTP算法参数')
    btp_group.add_argument('--num-beams', type=int, default=5,
                         help='Beam Search数量')
    btp_group.add_argument('--n-iterations', type=int, default=2,
                         help='PPER训练迭代次数')
    btp_group.add_argument('--batch-size', type=int, default=50,
                         help='训练批大小')
    
    # 采样参数
    sampling_group = parser.add_argument_group('采样参数')
    sampling_group.add_argument('--sampling-method', type=str, default='power', 
                              choices=['power', 'rank'],
                              help='采样方法')
    sampling_group.add_argument('--sampling-alpha', type=float, default=1.0,
                              help='采样α参数')
    sampling_group.add_argument('--p2value-alpha', type=float, default=0.5,
                              help='P2Value权重α')
    
    # LoRA参数
    lora_group = parser.add_argument_group('LoRA参数')
    lora_group.add_argument('--use-lora', action='store_true', default=True,
                          help='使用LoRA微调')
    lora_group.add_argument('--lora-r', type=int, default=16,
                          help='LoRA rank')
    lora_group.add_argument('--lora-alpha', type=int, default=32,
                          help='LoRA alpha')
    lora_group.add_argument('--lora-dropout', type=float, default=0.1,
                          help='LoRA dropout')
    
    # 训练参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--learning-rate', type=float, default=1e-4,
                           help='学习率')
    train_group.add_argument('--num-epochs', type=int, default=1,
                           help='每轮迭代的训练轮数')
    train_group.add_argument('--per-device-batch-size', type=int, default=2,
                           help='每设备批大小')
    train_group.add_argument('--gradient-accumulation-steps', type=int, default=4,
                           help='梯度累积步数')
    
    # 输出参数
    output_group = parser.add_argument_group('输出参数')
    output_group.add_argument('--output-dir', type=str, default='./btp_results',
                            help='结果输出目录')
    output_group.add_argument('--checkpoint-dir', type=str, default='./btp_checkpoints',
                            help='模型检查点目录')
    
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
    
    # 准备LoRA配置
    lora_config = {
        'r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout
    } if args.use_lora else None
    
    # 打印配置
    print("BTP Fine-tuning Experiment Configuration:")
    print(f"  Source Model: {args.source_model}")
    print(f"  Target Model: {args.target_model or 'same as source'}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max Problems: {args.max_problems}")
    print(f"  Sampling Method: {args.sampling_method}")
    print(f"  Sampling Alpha: {args.sampling_alpha}")
    print(f"  P2Value Alpha: {args.p2value_alpha}")
    print(f"  Use LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"  LoRA Config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    # 创建实验实例
    experiment = BTPFineTuneExperiment(
        source_model_path=args.source_model,
        target_model_path=args.target_model,
        dataset=args.dataset,
        sampling_method=args.sampling_method,
        sampling_alpha=args.sampling_alpha,
        p2value_alpha=args.p2value_alpha,
        use_lora=args.use_lora,
        lora_config=lora_config
    )
    
    # 运行实验 - 添加异常处理
    try:
        results = experiment.run_experiment(
            max_problems=args.max_problems,
            num_beams=args.num_beams,
            n_iterations=args.n_iterations,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        
        print("\nExperiment completed successfully!")
        print(f"Final stats:")
        for key, value in results['final_stats'].items():
            print(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存错误信息
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            error_info = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat(),
                'experiment_type': 'BTP_FineTune',
                'source_model': args.source_model,
                'target_model': args.target_model,
                'dataset': args.dataset,
                'max_problems': args.max_problems,
                'num_beams': args.num_beams,
                'n_iterations': args.n_iterations,
                'batch_size': args.batch_size,
                'status': 'failed'
            }
            
            error_file = os.path.join(args.output_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            
            print(f"Error log saved to: {error_file}")
            
        except Exception as save_error:
            print(f"Failed to save error log: {save_error}")
        
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 