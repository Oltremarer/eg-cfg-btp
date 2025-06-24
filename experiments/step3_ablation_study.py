#!/usr/bin/env python3
"""
实验三：消融研究 (Ablation Studies)
对比不同的经验回放策略
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
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from step2_btp_experiment import ExperienceReplayBuffer
from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests
from eg_cfg.model_utils import setup_device, load_model, load_tokenizer


class RandomReplayBuffer(ExperienceReplayBuffer):
    """随机采样的经验回放缓冲区"""
    
    def sample_prioritized(self, batch_size, failed_only=True):
        """重写为随机采样"""
        if failed_only:
            available = [exp for exp in self.buffer if exp['pass_rate_score'] < 1.0]
        else:
            available = list(self.buffer)
        
        if not available:
            return []
        
        # 随机采样
        sample_size = min(batch_size, len(available))
        return random.sample(available, sample_size)


class AblationExperiment:
    """消融实验主类"""
    
    def __init__(self, model_name: str, dataset: str = "mbpp"):
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
        
        print(f"Loaded {len(self.problems)} problems")
    
    def run_baseline_experiment(self, problems_subset, num_samples=3):
        """运行基线实验（无回放）"""
        print("Running baseline experiment...")
        
        results = []
        for task_id, problem in tqdm(problems_subset, desc="Baseline"):
            try:
                solutions = self.generate_simple_solutions(problem, num_samples)
                tested_solutions = self.test_solutions(problem, solutions)
                
                if tested_solutions:
                    best_result = max(tested_solutions, key=lambda x: x['pass_rate'])
                    
                    results.append({
                        'task_id': task_id,
                        'best_pass_rate': best_result['pass_rate'],
                        'fully_passed': best_result['fully_passed']
                    })
                else:
                    results.append({
                        'task_id': task_id,
                        'best_pass_rate': 0.0,
                        'fully_passed': False
                    })
                
            except Exception as e:
                print(f"Error in baseline for {task_id}: {e}")
                results.append({
                    'task_id': task_id,
                    'best_pass_rate': 0.0,
                    'fully_passed': False,
                    'error': str(e)
                })
        
        return results
    
    def generate_simple_solutions(self, problem, num_samples):
        """生成简单解决方案"""
        prompt = self.format_prompt(problem)
        solutions = []
        
        for i in range(num_samples):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.8,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                code = generated_text[len(prompt):].strip()
                
                solutions.append({
                    'code': code,
                    'method': 'simple_sampling'
                })
                
            except Exception as e:
                print(f"Error generating solution {i}: {e}")
                continue
        
        return solutions
    
    def test_solutions(self, problem, solutions):
        """测试解决方案"""
        results = []
        
        for sol in solutions:
            try:
                test_results = run_tests(sol['code'], problem['test_list'])
                passed_tests = sum(1 for r in test_results.values() if r.get('result', False))
                total_tests = len(test_results)
                
                results.append({
                    **sol,
                    'passed_tests': passed_tests,
                    'total_tests': total_tests,
                    'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                    'fully_passed': passed_tests == total_tests and total_tests > 0
                })
                
            except Exception as e:
                results.append({
                    **sol,
                    'passed_tests': 0,
                    'total_tests': len(problem.get('test_list', [])),
                    'pass_rate': 0.0,
                    'fully_passed': False,
                    'error': str(e)
                })
        
        return results
    
    def format_prompt(self, problem):
        """格式化提示词"""
        return f"""Problem:
{problem['text']}

Please provide a Python solution:

```python
"""
    
    def run_complete_ablation_study(self, output_dir, num_problems=20):
        """运行完整消融研究"""
        print("Starting Ablation Study...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 选择问题子集
        problems_list = list(self.problems.items())[:num_problems]
        
        # 运行基线实验
        baseline_results = self.run_baseline_experiment(problems_list)
        
        # 分析结果
        analysis = self.analyze_results(baseline_results)
        
        # 保存结果
        results = {
            'experiment_config': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'num_problems': num_problems,
                'timestamp': datetime.now().isoformat()
            },
            'baseline_results': baseline_results,
            'analysis': analysis
        }
        
        with open(output_path / "ablation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Ablation study completed!")
        return results
    
    def analyze_results(self, results):
        """分析结果"""
        if not results:
            return {}
        
        total_pass_rate = sum(r.get('best_pass_rate', 0) for r in results)
        success_count = sum(1 for r in results if r.get('fully_passed', False))
        
        return {
            'avg_pass_rate': total_pass_rate / len(results),
            'success_rate': success_count / len(results),
            'total_problems': len(results)
        }


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--output_dir", type=str, default="experiments/results/ablation")
    parser.add_argument("--num_problems", type=int, default=20)
    
    args = parser.parse_args()
    
    experiment = AblationExperiment(args.model_name, args.dataset)
    results = experiment.run_complete_ablation_study(args.output_dir, args.num_problems)
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 