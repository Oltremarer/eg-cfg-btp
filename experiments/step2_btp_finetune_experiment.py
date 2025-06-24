#!/usr/bin/env python3
"""
BTP (Beam Search + Testing + Prioritized Experience Replay) 微调实验
支持论文中的完整BTP算法，包括真正的模型微调
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
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install transformers peft datasets")
    sys.exit(1)

# 项目相关导入
from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests


class P2ValueCalculator:
    """P2Value计算器"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def calculate_p2value(self, possibility: float, pass_rate: float) -> float:
        """计算P2Value = α × possibility + (1-α) × pass_rate"""
        return self.alpha * possibility + (1 - self.alpha) * pass_rate


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
    
    def get_stats(self) -> Dict:
        """获取缓冲区统计信息"""
        if not self.buffer:
            return {}
        
        experiences = list(self.buffer)
        p2values = [exp['p2value'] for exp in experiences]
        pass_rates = [exp['pass_rate'] for exp in experiences]
        
        return {
            'total_experiences': len(experiences),
            'avg_p2value': np.mean(p2values),
            'std_p2value': np.std(p2values),
            'avg_pass_rate': np.mean(pass_rates),
            'fully_passed_count': sum(1 for exp in experiences if exp['pass_rate'] >= 1.0)
        }


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
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.source_model_path)
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载目标模型（用于微调）
        if self.target_model_path != self.source_model_path:
            print(f"Loading target model: {self.target_model_path}")
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.target_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
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
        
        # 分词
        def tokenize_function(examples):
            return self.model_manager.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=1024
            )
        
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
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
        initial_stats = self.experience_buffer.get_stats()
        print(f"\nInitial experience buffer stats:")
        for key, value in initial_stats.items():
            print(f"  {key}: {value}")
        
        # 阶段2: PPER微调
        self.phase2_pper_training(n_iterations, batch_size)
        
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
            'final_stats': self.experience_buffer.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, f"btp_finetune_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nExperiment results saved to: {result_file}")
        return results


def main():
    parser = argparse.ArgumentParser(description='BTP Fine-tuning Experiment')
    
    # 模型相关参数
    parser.add_argument('--source-model', type=str, required=True,
                       help='Source model path for initial generation')
    parser.add_argument('--target-model', type=str, default=None,
                       help='Target model path for fine-tuning (default: same as source)')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='mbpp',
                       help='Dataset name (default: mbpp)')
    parser.add_argument('--max-problems', type=int, default=100,
                       help='Maximum number of problems to process')
    
    # BTP算法参数
    parser.add_argument('--num-beams', type=int, default=5,
                       help='Number of beams for beam search')
    parser.add_argument('--n-iterations', type=int, default=3,
                       help='Number of PPER training iterations')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for training')
    
    # 采样参数
    parser.add_argument('--sampling-method', type=str, default='power', 
                       choices=['power', 'rank'],
                       help='Sampling method: power or rank')
    parser.add_argument('--sampling-alpha', type=float, default=1.0,
                       help='Alpha parameter for sampling')
    parser.add_argument('--p2value-alpha', type=float, default=0.5,
                       help='Alpha parameter for P2Value calculation')
    
    # LoRA参数
    parser.add_argument('--use-lora', action='store_true', default=True,
                       help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora-r', type=int, default=64,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=128,
                       help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # 训练参数
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of training epochs per iteration')
    parser.add_argument('--per-device-batch-size', type=int, default=4,
                       help='Per device batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='./btp_results',
                       help='Output directory for results')
    parser.add_argument('--checkpoint-dir', type=str, default='./btp_checkpoints',
                       help='Directory for model checkpoints')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
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
    
    # 运行实验
    results = experiment.run_experiment(
        max_problems=args.max_problems,
        num_beams=args.num_beams,
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    print("\nExperiment completed!")
    print(f"Final stats:")
    for key, value in results['final_stats'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 