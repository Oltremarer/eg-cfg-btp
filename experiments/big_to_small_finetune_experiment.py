#!/usr/bin/env python3
"""
本地大模型采样 -> 本地小模型微调的BTP实验
专门设计用于本地模型的微调实验，无需API调用
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional
import logging
import gc

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
        GenerationConfig
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
    """P2Value计算器：平衡可能性和通过率"""
    
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
    
    def calculate_p2value(self, possibility: float, pass_rate: float) -> float:
        """计算P2Value = α × pass_rate + (1-α) × possibility"""
        return self.alpha * pass_rate + (1 - self.alpha) * possibility


class PrioritizedSampler:
    """优先经验采样器"""
    
    def __init__(self, sampling_method: str = "power", alpha: float = 1.5):
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
        p2values = np.maximum(p2values, 1e-8)  # 避免除零
        
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
        sorted_experiences = sorted(experiences, key=lambda x: x['p2value'], reverse=True)
        
        ranks = np.arange(1, len(sorted_experiences) + 1)
        probabilities = 1.0 / ranks
        probabilities = probabilities / np.sum(probabilities)
        
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
        return list(self.buffer)
    
    def get_stats(self) -> Dict:
        """获取缓冲区统计信息"""
        if not self.buffer:
            return {'total_experiences': 0}
        
        experiences = list(self.buffer)
        p2values = [exp['p2value'] for exp in experiences]
        pass_rates = [exp['pass_rate'] for exp in experiences]
        
        return {
            'total_experiences': len(experiences),
            'avg_p2value': np.mean(p2values),
            'std_p2value': np.std(p2values),
            'avg_pass_rate': np.mean(pass_rates),
            'perfect_solutions': sum(1 for exp in experiences if exp['pass_rate'] >= 1.0),
            'zero_pass_solutions': sum(1 for exp in experiences if exp['pass_rate'] == 0.0)
        }


class LocalModelManager:
    """本地模型管理器 - 专门处理本地大小模型"""
    
    def __init__(self, 
                 source_model_path: str,
                 target_model_path: str,
                 use_lora: bool = True,
                 lora_config: Optional[Dict] = None):
        
        self.source_model_path = source_model_path
        self.target_model_path = target_model_path
        self.use_lora = use_lora
        
        print("🚀 初始化本地模型管理器")
        print(f"📊 源模型（采样）: {source_model_path}")
        print(f"🎯 目标模型（微调）: {target_model_path}")
        
        # 加载tokenizer（共享）
        self.tokenizer = AutoTokenizer.from_pretrained(
            source_model_path, 
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载源模型（用于采样）
        self._load_source_model()
        
        # 加载目标模型（用于微调）
        self._load_target_model(lora_config)
    
    def _load_source_model(self):
        """加载源模型用于采样"""
        print(f"🔄 加载源模型: {self.source_model_path}")
        
        self.source_model = AutoModelForCausalLM.from_pretrained(
            self.source_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 设置生成配置
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=512,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print(f"✅ 源模型加载完成，显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    
    def _load_target_model(self, lora_config: Optional[Dict]):
        """加载目标模型用于微调"""
        print(f"🔄 加载目标模型: {self.target_model_path}")
        
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 配置LoRA
        if self.use_lora:
            lora_config = lora_config or {
                'r': 32,
                'lora_alpha': 64,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                'lora_dropout': 0.05,
                'bias': 'none',
                'task_type': TaskType.CAUSAL_LM
            }
            
            peft_config = LoraConfig(**lora_config)
            self.target_model = get_peft_model(self.target_model, peft_config)
            self.target_model.print_trainable_parameters()
        
        print(f"✅ 目标模型加载完成，总显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    
    def generate_beam_candidates(self, 
                               prompt: str, 
                               num_beams: int = 8, 
                               max_new_tokens: int = 512) -> List[Dict]:
        """使用源模型生成beam search候选"""
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.source_model.device) for k, v in inputs.items()}
        
        # 生成配置
        generation_config = GenerationConfig(
            do_sample=True,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        # 生成
        with torch.no_grad():
            outputs = self.source_model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # 解析结果
        candidates = []
        sequences = outputs.sequences
        scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else None
        
        for i in range(num_beams):
            # 解码生成的序列
            generated_sequence = sequences[i]
            generated_text = self.tokenizer.decode(
                generated_sequence[inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # 提取代码
            code = self._extract_code_from_text(generated_text)
            
            # 计算可能性（使用分数或默认值）
            possibility = float(scores[i]) if scores is not None else 1.0 / (i + 1)
            
            candidates.append({
                'code': code,
                'raw_text': generated_text,
                'possibility': possibility,
                'beam_rank': i
            })
        
        return candidates
    
    def _extract_code_from_text(self, text: str) -> str:
        """从生成的文本中提取Python代码"""
        import re
        
        # 尝试提取```python代码块
        python_pattern = r'```python\n(.*?)```'
        match = re.search(python_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 尝试提取```代码块  
        code_pattern = r'```\n(.*?)```'
        match = re.search(code_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 如果没有代码块，返回整个文本
        return text.strip()
    
    def clear_source_model_cache(self):
        """清理源模型缓存以节省显存"""
        if hasattr(self, 'source_model'):
            del self.source_model
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 源模型缓存已清理")


class LocalBTPFineTuneExperiment:
    """本地BTP微调实验主类"""
    
    def __init__(self, 
                 source_model_path: str,
                 target_model_path: str,
                 dataset: str = "mbpp",
                 sampling_method: str = "power",
                 sampling_alpha: float = 1.5,
                 p2value_alpha: float = 0.7,
                 use_lora: bool = True,
                 lora_config: Optional[Dict] = None):
        
        self.dataset = dataset
        
        # 初始化组件
        self.p2calculator = P2ValueCalculator(p2value_alpha)
        self.sampler = PrioritizedSampler(sampling_method, sampling_alpha)
        self.experience_buffer = ExperienceReplayBuffer()
        
        # 加载数据集
        if dataset == "mbpp":
            self.problems = load_mbpp_problems()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # 初始化模型管理器
        self.model_manager = LocalModelManager(
            source_model_path, 
            target_model_path,
            use_lora,
            lora_config
        )
        
        print("=" * 60)
        print("🚀 本地BTP微调实验初始化完成")
        print(f"📊 采样方法: {sampling_method} (α={sampling_alpha})")
        print(f"🎯 P2Value权重: pass_rate={p2value_alpha}, possibility={1-p2value_alpha}")
        print(f"🔧 使用LoRA: {use_lora}")
        print("=" * 60)
    
    def format_problem_prompt(self, problem: Dict) -> str:
        """格式化问题为模型输入"""
        prompt = f"""Solve the following programming problem:

{problem['text']}

Please provide a complete Python function that solves this problem.

```python
"""
        return prompt
    
    def phase1_beam_search_sampling(self, problems: List[Dict], num_beams: int = 8) -> None:
        """阶段1: 使用大模型进行Beam Search采样"""
        print("=" * 40)
        print("📊 阶段1: Beam Search采样")
        print("=" * 40)
        
        for problem_id, problem in tqdm(problems, desc="🔍 采样代码候选"):
            prompt = self.format_problem_prompt(problem)
            
            try:
                # 生成候选
                candidates = self.model_manager.generate_beam_candidates(
                    prompt, num_beams=num_beams
                )
                
                # 测试每个候选
                for candidate in candidates:
                    try:
                        # 运行测试
                        test_results = run_tests(candidate['code'], problem['test_list'])
                        passed_tests = sum(1 for r in test_results.values() if r.get('result', False))
                        total_tests = len(test_results)
                        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
                        
                        # 创建经验
                        experience = {
                            'problem_id': problem_id,
                            'problem_text': problem['text'],
                            'code': candidate['code'],
                            'possibility': candidate['possibility'],
                            'pass_rate': pass_rate,
                            'test_results': test_results,
                            'beam_rank': candidate['beam_rank'],
                            'passed_tests': passed_tests,
                            'total_tests': total_tests
                        }
                        
                        self.experience_buffer.add_experience(experience)
                        
                    except Exception as e:
                        # 测试失败，记录为0通过率
                        experience = {
                            'problem_id': problem_id,
                            'problem_text': problem['text'],
                            'code': candidate['code'],
                            'possibility': candidate['possibility'],
                            'pass_rate': 0.0,
                            'error': str(e),
                            'beam_rank': candidate['beam_rank'],
                            'passed_tests': 0,
                            'total_tests': len(problem['test_list'])
                        }
                        self.experience_buffer.add_experience(experience)
                        
            except Exception as e:
                print(f"❌ 问题 {problem_id} 生成失败: {e}")
                continue
        
        # 显示采样统计
        stats = self.experience_buffer.get_stats()
        print("\n📈 采样统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def phase2_pper_training(self, 
                           n_iterations: int = 3,
                           batch_size: int = 32,
                           training_args: Optional[TrainingArguments] = None) -> None:
        """阶段2: 优先经验回放训练"""
        print("\n" + "=" * 40)
        print("🎯 阶段2: PPER微调训练")
        print("=" * 40)
        
        # 清理源模型以节省显存
        self.model_manager.clear_source_model_cache()
        
        for iteration in range(n_iterations):
            print(f"\n🔄 迭代 {iteration + 1}/{n_iterations}")
            
            # 优先采样
            all_experiences = self.experience_buffer.get_all_experiences()
            sampled_experiences = self.sampler.sample(all_experiences, batch_size)
            
            print(f"📦 采样了 {len(sampled_experiences)} 个高质量经验")
            
            # 显示采样质量
            avg_pass_rate = np.mean([exp['pass_rate'] for exp in sampled_experiences])
            avg_p2value = np.mean([exp['p2value'] for exp in sampled_experiences])
            print(f"📊 平均通过率: {avg_pass_rate:.3f}, 平均P2Value: {avg_p2value:.3f}")
            
            # 准备训练数据
            train_dataset = self._prepare_training_dataset(sampled_experiences)
            
            # 执行微调
            self._finetune_model(train_dataset, training_args, iteration)
    
    def _prepare_training_dataset(self, experiences: List[Dict]) -> Dataset:
        """准备微调训练数据集"""
        texts = []
        
        for exp in experiences:
            # 指令微调格式
            instruction = f"Solve this programming problem:\n{exp['problem_text']}"
            response = exp['code']
            
            # 格式化训练文本
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{self.model_manager.tokenizer.eos_token}"
            texts.append(text)
        
        # 创建数据集
        dataset = Dataset.from_dict({'text': texts})
        
        # 分词
        def tokenize_function(examples):
            tokenized = self.model_manager.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=['text']
        )
        
        return tokenized_dataset
    
    def _finetune_model(self, 
                       train_dataset: Dataset, 
                       training_args: Optional[TrainingArguments],
                       iteration: int) -> None:
        """执行模型微调"""
        
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=f"./local_btp_checkpoints/iteration_{iteration}",
                num_train_epochs=2,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                warmup_steps=50,
                learning_rate=1e-4,
                fp16=True,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                remove_unused_columns=False,
                dataloader_num_workers=0,
                report_to=None
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
        print(f"🚀 开始微调迭代 {iteration}...")
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        print(f"💾 模型已保存到 {training_args.output_dir}")
    
    def run_experiment(self, 
                      max_problems: int = 50,
                      num_beams: int = 8,
                      n_iterations: int = 3,
                      batch_size: int = 32,
                      output_dir: str = "./local_btp_results") -> Dict:
        """运行完整的BTP实验"""
        
        print("=" * 80)
        print("🚀 本地BTP微调实验开始")
        print("=" * 80)
        
        # 选择问题子集
        problems_list = list(self.problems.items())[:max_problems]
        print(f"📋 处理 {len(problems_list)} 个编程问题")
        
        # 阶段1: Beam Search采样
        self.phase1_beam_search_sampling(problems_list, num_beams)
        
        # 显示初始统计
        initial_stats = self.experience_buffer.get_stats()
        print(f"\n📊 初始经验统计:")
        for key, value in initial_stats.items():
            print(f"  {key}: {value}")
        
        # 阶段2: PPER微调
        self.phase2_pper_training(n_iterations, batch_size)
        
        # 保存实验结果
        results = {
            'experiment_type': 'Local_BTP_FineTune',
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(output_dir, f"local_btp_results_{timestamp}.json")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 实验结果已保存到: {result_file}")
        print("=" * 80)
        print("✅ 本地BTP微调实验完成!")
        print("=" * 80)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='本地大模型->小模型BTP微调实验')
    
    # 模型参数
    parser.add_argument('--source-model', type=str, 
                       default='deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
                       help='源模型路径（用于采样）')
    parser.add_argument('--target-model', type=str, 
                       default='deepseek-ai/deepseek-coder-1.3b-instruct',
                       help='目标模型路径（用于微调）')
    
    # 实验参数
    parser.add_argument('--max-problems', type=int, default=50,
                       help='最大问题数量')
    parser.add_argument('--num-beams', type=int, default=8,
                       help='Beam Search候选数量')
    parser.add_argument('--n-iterations', type=int, default=3,
                       help='微调迭代次数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='每次微调的批次大小')
    
    # 采样参数
    parser.add_argument('--sampling-method', type=str, default='power', 
                       choices=['power', 'rank'],
                       help='采样方法')
    parser.add_argument('--sampling-alpha', type=float, default=1.5,
                       help='采样α参数')
    parser.add_argument('--p2value-alpha', type=float, default=0.7,
                       help='P2Value α参数')
    
    # LoRA参数
    parser.add_argument('--lora-r', type=int, default=32,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=64,
                       help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                       help='LoRA dropout')
    
    # 训练参数
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--num-epochs', type=int, default=2,
                       help='每次迭代的训练轮数')
    parser.add_argument('--per-device-batch-size', type=int, default=2,
                       help='每设备批次大小')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                       help='梯度累积步数')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='./local_btp_results',
                       help='结果输出目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # LoRA配置
    lora_config = {
        'r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'bias': 'none',
        'task_type': 'CAUSAL_LM'
    }
    
    # 打印配置
    print("🔧 本地BTP微调实验配置:")
    print(f"  📊 源模型: {args.source_model}")
    print(f"  🎯 目标模型: {args.target_model}")
    print(f"  📋 问题数量: {args.max_problems}")
    print(f"  🔍 Beam数量: {args.num_beams}")
    print(f"  🔄 迭代次数: {args.n_iterations}")
    print(f"  📦 批次大小: {args.batch_size}")
    print(f"  🎲 采样方法: {args.sampling_method} (α={args.sampling_alpha})")
    print(f"  ⚖️ P2Value权重: {args.p2value_alpha}")
    print(f"  🔧 LoRA: r={args.lora_r}, α={args.lora_alpha}, dropout={args.lora_dropout}")
    print()
    
    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/checkpoints",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=None
    )
    
    # 创建实验
    experiment = LocalBTPFineTuneExperiment(
        source_model_path=args.source_model,
        target_model_path=args.target_model,
        dataset="mbpp",
        sampling_method=args.sampling_method,
        sampling_alpha=args.sampling_alpha,
        p2value_alpha=args.p2value_alpha,
        use_lora=True,
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
    
    print("🎉 实验完成!")
    final_stats = results['final_stats']
    print(f"📈 最终统计:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 