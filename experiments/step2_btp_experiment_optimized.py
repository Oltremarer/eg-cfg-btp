#!/usr/bin/env python3
"""
实验二优化版：快速BTP实验（适合CPU运行）
使用更小的模型和减少的beam search
"""

import os
import sys
import json
import torch
import numpy as np
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
from datetime import datetime
import math

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests
from eg_cfg.model_utils import setup_device, load_model, load_tokenizer


class P2ValueCalculator:
    """计算P2Value指标：综合生成概率和测试通过率"""
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    
    def calculate_p2value(self, log_prob, sequence_length, passed_tests, total_tests):
        """计算P2Value指标"""
        # 标准化概率分数
        possibility = math.exp(log_prob / max(sequence_length, 1))
        possibility = min(possibility, 1.0)
        
        # 计算通过率
        pass_rate = passed_tests / max(total_tests, 1)
        
        # 综合分数
        p2value = self.alpha * possibility + (1 - self.alpha) * pass_rate
        return p2value, possibility, pass_rate


class ExperienceReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.p2calculator = P2ValueCalculator()
    
    def add_experience(self, experience):
        """添加经验到缓冲区"""
        p2value, possibility, pass_rate = self.p2calculator.calculate_p2value(
            experience['log_prob'],
            experience['sequence_length'],
            experience['passed_tests'],
            experience['total_tests']
        )
        
        experience['p2value'] = p2value
        experience['possibility_score'] = possibility
        experience['pass_rate_score'] = pass_rate
        
        self.buffer.append(experience)
    
    def sample_prioritized(self, batch_size, failed_only=True):
        """基于P2Value优先采样"""
        if failed_only:
            available = [exp for exp in self.buffer if exp['pass_rate_score'] < 1.0]
        else:
            available = list(self.buffer)
        
        if not available:
            return []
        
        # 按P2Value排序
        available.sort(key=lambda x: x['p2value'], reverse=True)
        return available[:min(batch_size, len(available))]


class BTPExperimentOptimized:
    """优化的BTP实验类（CPU友好）"""
    
    def __init__(self, model_name: str, dataset: str):
        self.model_name = model_name
        self.dataset = dataset
        
        print(f"Loading model: {model_name}")
        self.device = setup_device()
        
        # 强制使用CPU以避免内存问题
        if "cuda" in str(self.device):
            print("检测到CUDA，但为了稳定性强制使用CPU")
            self.device = torch.device("cpu")
        
        self.model, self.tokenizer = load_model(model_name, self.device)
        
        # 设置模型为评估模式并优化
        self.model.eval()
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False  # 减少内存使用
        
        print(f"Loading dataset: {dataset}")
        if dataset == "mbpp":
            self.problems = load_mbpp_problems()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        self.replay_buffer = ExperienceReplayBuffer()
        print(f"Loaded {len(self.problems)} problems")
    
    def simple_generation(self, prompt, num_samples=3, max_tokens=128):
        """简化的代码生成（不使用beam search）"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        results = []
        
        # 生成多个样本
        for i in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # 获取生成的文本
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = generated_text[len(prompt):].strip()
            
            # 估算log概率（简化版）
            sequence_length = outputs[0].shape[0] - inputs['input_ids'].shape[1]
            log_prob = -sequence_length * 0.1  # 简化的概率估算
            
            results.append({
                'code': code,
                'log_prob': log_prob,
                'sequence_length': sequence_length,
                'sample_rank': i + 1
            })
            
            print(f"    生成样本 {i+1}: {code[:50]}...")
        
        return results
    
    def test_code_solutions(self, problem, solutions):
        """测试代码解决方案"""
        results = []
        
        for sol in solutions:
            try:
                # 限制测试时间
                test_results = run_tests(sol['code'], problem['test_list'][:2])  # 只测试前2个用例
                passed_tests = sum(1 for r in test_results.values() if r.get('result', False))
                total_tests = len(test_results)
                
                results.append({
                    **sol,
                    'test_results': test_results,
                    'passed_tests': passed_tests,
                    'total_tests': total_tests,
                    'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                    'fully_passed': passed_tests == total_tests and total_tests > 0
                })
                
            except Exception as e:
                results.append({
                    **sol,
                    'test_results': {},
                    'passed_tests': 0,
                    'total_tests': len(problem.get('test_list', [])[:2]),
                    'pass_rate': 0.0,
                    'fully_passed': False,
                    'error': str(e)
                })
        
        return results
    
    def format_prompt(self, problem):
        """格式化问题为提示词"""
        return f"""解决这个编程问题：

{problem['text']}

```python
"""
    
    def collect_experiences(self, max_problems=5, num_samples=3):
        """收集经验数据（优化版）"""
        print("开始收集经验数据...")
        
        problems_list = list(self.problems.items())[:max_problems]
        
        for task_id, problem in tqdm(problems_list, desc="处理问题"):
            try:
                print(f"\n处理问题 {task_id}: {problem['text'][:50]}...")
                
                prompt = self.format_prompt(problem)
                
                # 简化生成
                generation_results = self.simple_generation(prompt, num_samples=num_samples)
                
                # 测试代码
                tested_results = self.test_code_solutions(problem, generation_results)
                
                # 添加到回放缓冲区
                for result in tested_results:
                    experience = {
                        'task_id': task_id,
                        'code': result['code'],
                        'log_prob': result['log_prob'],
                        'sequence_length': result['sequence_length'],
                        'passed_tests': result['passed_tests'],
                        'total_tests': result['total_tests'],
                        'pass_rate': result['pass_rate'],
                        'fully_passed': result['fully_passed']
                    }
                    self.replay_buffer.add_experience(experience)
                
                print(f"  问题 {task_id} 完成，通过率：{[r['pass_rate'] for r in tested_results]}")
                
            except Exception as e:
                print(f"处理问题 {task_id} 时出错: {e}")
                continue
        
        print(f"收集完成！总共 {len(self.replay_buffer.buffer)} 个经验")
    
    def run_experiment(self, output_dir, collect_problems=5):
        """运行优化的BTP实验"""
        print("开始BTP实验（优化版）...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 收集经验
        self.collect_experiences(max_problems=collect_problems)
        
        # 分析结果
        analysis = self.analyze_buffer()
        
        # 保存结果
        results = {
            'experiment_config': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'collect_problems': collect_problems,
                'optimization': 'CPU友好版本',
                'timestamp': datetime.now().isoformat()
            },
            'analysis': analysis,
            'experiences_count': len(self.replay_buffer.buffer)
        }
        
        with open(output_path / "btp_optimized_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*50)
        print("BTP实验完成！")
        print(f"处理了 {collect_problems} 个问题")
        print(f"收集了 {len(self.replay_buffer.buffer)} 个经验")
        print(f"结果保存到: {output_path}")
        
        return results
    
    def analyze_buffer(self):
        """分析缓冲区"""
        if not self.replay_buffer.buffer:
            return {}
        
        experiences = list(self.replay_buffer.buffer)
        p2values = [exp['p2value'] for exp in experiences]
        pass_rates = [exp['pass_rate'] for exp in experiences]
        
        return {
            'total_experiences': len(experiences),
            'successful_experiences': len([e for e in experiences if e['fully_passed']]),
            'failed_experiences': len([e for e in experiences if not e['fully_passed']]),
            'avg_p2value': np.mean(p2values),
            'p2value_std': np.std(p2values),
            'avg_pass_rate': np.mean(pass_rates),
            'success_rate': len([e for e in experiences if e['fully_passed']]) / len(experiences)
        }


def main():
    parser = argparse.ArgumentParser(description="优化的BTP实验")
    parser.add_argument("--model_name", type=str, 
                       default="HuggingFaceTB/SmolLM2-135M-Instruct",
                       help="模型名称（推荐SmolLM2-135M-Instruct）")
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--output_dir", type=str, default="experiments/results/btp_optimized")
    parser.add_argument("--collect_problems", type=int, default=5,
                       help="处理的问题数量（建议5-10个）")
    
    args = parser.parse_args()
    
    print(f"开始优化的BTP实验")
    print(f"模型: {args.model_name}")
    print(f"问题数量: {args.collect_problems}")
    
    experiment = BTPExperimentOptimized(args.model_name, args.dataset)
    results = experiment.run_experiment(args.output_dir, args.collect_problems)
    
    print(f"\n结果保存到: {args.output_dir}")


if __name__ == "__main__":
    main() 