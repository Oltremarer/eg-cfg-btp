#!/usr/bin/env python3
"""
实验二：核心BTP管道有效性验证
BTP = Beam Search + Testing + Prioritized Experience Replay
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


class BTPExperiment:
    """BTP实验主类"""
    
    def __init__(self, model_name: str, dataset: str):
        self.model_name = model_name
        self.dataset = dataset
        
        print(f"Loading model: {model_name}")
        self.device = setup_device()
        self.model, self.tokenizer = load_model(model_name, self.device)
        
        print(f"Loading dataset: {dataset}")
        if dataset == "mbpp":
            self.problems = load_mbpp_problems()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        self.replay_buffer = ExperienceReplayBuffer()
        print(f"Loaded {len(self.problems)} problems")
    
    def beam_search_with_probabilities(self, prompt, num_beams=5, max_tokens=512):
        """使用beam search生成代码并计算概率"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        sequences = outputs.sequences
        scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else None
        
        for i, sequence in enumerate(sequences):
            generated_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            code = generated_text[len(prompt):].strip()
            
            if scores is not None:
                log_prob = scores[i].item()
            else:
                log_prob = -10.0  # fallback value
            
            results.append({
                'code': code,
                'log_prob': log_prob,
                'sequence_length': len(sequence) - inputs['input_ids'].shape[1],
                'beam_rank': i + 1
            })
        
        return results
    
    def test_code_solutions(self, problem, solutions):
        """测试代码解决方案"""
        results = []
        
        for sol in solutions:
            try:
                test_results = run_tests(sol['code'], problem['test_list'])
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
                    'total_tests': len(problem.get('test_list', [])),
                    'pass_rate': 0.0,
                    'fully_passed': False,
                    'error': str(e)
                })
        
        return results
    
    def format_prompt(self, problem):
        """格式化问题为提示词"""
        return f"""Problem:
{problem['text']}

Please provide a Python solution:

```python
"""
    
    def collect_experiences(self, max_problems=100, num_beams=5):
        """收集经验数据"""
        print(f"开始收集经验数据... (问题数量: {max_problems}, Beam数量: {num_beams})")
        
        problems_list = list(self.problems.items())[:max_problems]
        
        for task_id, problem in tqdm(problems_list, desc="处理问题"):
            try:
                print(f"\n处理问题 {task_id}: {problem['text'][:50]}...")
                prompt = self.format_prompt(problem)
                
                # Beam search生成
                beam_results = self.beam_search_with_probabilities(prompt, num_beams=num_beams)
                
                # 测试代码
                tested_results = self.test_code_solutions(problem, beam_results)
                
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
                
            except Exception as e:
                print(f"Error processing {task_id}: {e}")
                continue
        
        print(f"Collected {len(self.replay_buffer.buffer)} experiences")
    
    def run_experiment(self, output_dir, collect_problems=100, num_beams=5):
        """运行BTP实验"""
        print("Starting BTP Experiment...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 收集经验
        self.collect_experiences(max_problems=collect_problems, num_beams=num_beams)
        
        # 分析结果
        analysis = self.analyze_buffer()
        
        # 保存详细结果
        detailed_experiences = []
        for exp in list(self.replay_buffer.buffer):
            detailed_experiences.append({
                'task_id': exp['task_id'],
                'code': exp['code'][:200],  # 只保存前200字符避免文件过大
                'fully_passed': exp['fully_passed'],
                'pass_rate': exp['pass_rate'],
                'p2value': exp['p2value'],
                'passed_tests': exp['passed_tests'],
                'total_tests': exp['total_tests']
            })
        
        results = {
            'experiment_config': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'collect_problems': collect_problems,
                'num_beams': num_beams,
                'timestamp': datetime.now().isoformat()
            },
            'analysis': analysis,
            'detailed_experiences': detailed_experiences
        }
        
        with open(output_path / "btp_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("BTP experiment completed!")
        return results
    
    def analyze_buffer(self):
        """分析缓冲区"""
        if not self.replay_buffer.buffer:
            return {}
        
        experiences = list(self.replay_buffer.buffer)
        p2values = [exp['p2value'] for exp in experiences]
        
        return {
            'total_experiences': len(experiences),
            'failed_experiences': len([e for e in experiences if not e['fully_passed']]),
            'avg_p2value': np.mean(p2values),
            'p2value_std': np.std(p2values)
        }


def main():
    # 预定义的模型选项
    AVAILABLE_MODELS = {
        "smollm-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "smollm-360m": "HuggingFaceTB/SmolLM2-360M-Instruct", 
        "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "deepseek-v2-lite": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    }
    
    parser = argparse.ArgumentParser(description="BTP Experiment - CPU优化版")
    parser.add_argument("--model_name", type=str, 
                       help=f"模型名称，可以使用简称：{list(AVAILABLE_MODELS.keys())} 或完整的HuggingFace模型路径")
    parser.add_argument("--model_preset", type=str, choices=list(AVAILABLE_MODELS.keys()),
                       default="smollm-135m", 
                       help="预设模型选项（推荐CPU使用：smollm-135m）")
    parser.add_argument("--dataset", type=str, default="mbpp", choices=["mbpp", "humaneval"])
    parser.add_argument("--output_dir", type=str, default="experiments/results/btp")
    parser.add_argument("--collect_problems", type=int, default=5,
                       help="处理的问题数量（CPU建议5-10个）")
    parser.add_argument("--num_beams", type=int, default=3,
                       help="Beam search数量（CPU建议2-3个）")
    parser.add_argument("--force_cpu", action="store_true",
                       help="强制使用CPU运行")
    
    args = parser.parse_args()
    
    # 确定使用的模型
    if args.model_name:
        if args.model_name in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[args.model_name]
            print(f"使用预设模型: {args.model_name} -> {model_name}")
        else:
            model_name = args.model_name
            print(f"使用自定义模型: {model_name}")
    else:
        model_name = AVAILABLE_MODELS[args.model_preset]
        print(f"使用默认模型: {args.model_preset} -> {model_name}")
    
    print(f"实验配置:")
    print(f"  模型: {model_name}")
    print(f"  数据集: {args.dataset}")
    print(f"  问题数量: {args.collect_problems}")
    print(f"  Beam数量: {args.num_beams}")
    print(f"  强制CPU: {args.force_cpu}")
    
    # 创建实验实例
    experiment = BTPExperiment(model_name, args.dataset)
    
    # 如果强制使用CPU
    if args.force_cpu:
        experiment.device = torch.device("cpu")
        experiment.model = experiment.model.to(experiment.device)
        print("已强制切换到CPU模式")
    
    # 运行实验
    results = experiment.run_experiment(args.output_dir, args.collect_problems, args.num_beams)
    
    print(f"\n结果保存到: {args.output_dir}")
    print("实验完成！")


if __name__ == "__main__":
    main() 