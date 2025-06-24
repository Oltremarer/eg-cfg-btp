#!/usr/bin/env python3
"""
实验二改进版：使用改进的提示词模板
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
import re

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests
from eg_cfg.model_utils import setup_device, load_model, load_tokenizer


class BTPExperimentImproved:
    """改进的BTP实验类"""
    
    def __init__(self, model_name: str, dataset: str):
        self.model_name = model_name
        self.dataset = dataset
        
        print(f"Loading model: {model_name}")
        self.device = setup_device()
        
        if "cuda" in str(self.device):
            print("检测到CUDA，但为了稳定性强制使用CPU")
            self.device = torch.device("cpu")
        
        self.model, self.tokenizer = load_model(model_name, self.device)
        self.model.eval()
        
        print(f"Loading dataset: {dataset}")
        if dataset == "mbpp":
            self.problems = load_mbpp_problems()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        print(f"Loaded {len(self.problems)} problems")
    
    def extract_function_name_from_tests(self, test_list):
        """从测试用例中提取函数名"""
        for test in test_list:
            if 'assert' in test:
                # 匹配 assert function_name(...) 模式
                match = re.search(r'assert\s+(\w+)\s*\(', test)
                if match:
                    return match.group(1)
        return "solution"  # 默认函数名
    
    def create_improved_prompt(self, problem):
        """创建改进的提示词"""
        # 提取函数名
        func_name = self.extract_function_name_from_tests(problem['test_list'])
        
        # 分析测试用例来推断参数
        sample_test = problem['test_list'][0] if problem['test_list'] else ""
        
        prompt = f"""请实现以下Python函数：

问题描述：{problem['text']}

函数名：{func_name}

测试用例：
{chr(10).join(problem['test_list'])}

请严格按照以下格式实现：

```python
def {func_name}(
"""
        return prompt, func_name
    
    def simple_generation(self, prompt, num_samples=3, max_tokens=200):
        """简化的代码生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        results = []
        
        for i in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = generated_text[len(prompt):].strip()
            
            # 后处理：确保代码块完整
            if '```' in code:
                code = code.split('```')[0]
            
            # 简单的代码修复
            code = self.post_process_code(code)
            
            sequence_length = outputs[0].shape[0] - inputs['input_ids'].shape[1]
            log_prob = -sequence_length * 0.1
            
            results.append({
                'code': code,
                'log_prob': log_prob,
                'sequence_length': sequence_length,
                'sample_rank': i + 1
            })
            
            print(f"    生成样本 {i+1}: {code[:100]}...")
        
        return results
    
    def post_process_code(self, code):
        """后处理生成的代码"""
        # 去掉常见的前后缀
        code = code.strip()
        
        # 如果没有缩进，添加基本缩进
        lines = code.split('\n')
        processed_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
                processed_lines.append(line.strip())
            elif in_function and line.strip():
                if not line.startswith('    ') and not line.startswith('\t'):
                    processed_lines.append('    ' + line.strip())
                else:
                    processed_lines.append(line)
            elif line.strip():
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
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
                    'fully_passed': passed_tests == total_tests and total_tests > 0,
                    'test_errors': [r.get('error', '') for r in test_results.values() if r.get('error')]
                })
                
            except Exception as e:
                results.append({
                    **sol,
                    'test_results': {},
                    'passed_tests': 0,
                    'total_tests': len(problem.get('test_list', [])),
                    'pass_rate': 0.0,
                    'fully_passed': False,
                    'error': str(e),
                    'test_errors': [str(e)]
                })
        
        return results
    
    def run_experiment(self, output_dir, collect_problems=3):
        """运行改进的实验"""
        print("开始改进的BTP实验...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        problems_list = list(self.problems.items())[:collect_problems]
        all_results = []
        
        for task_id, problem in tqdm(problems_list, desc="处理问题"):
            try:
                print(f"\n处理问题 {task_id}: {problem['text'][:50]}...")
                
                # 创建改进的提示词
                prompt, expected_func_name = self.create_improved_prompt(problem)
                print(f"期望函数名: {expected_func_name}")
                
                # 生成代码
                generation_results = self.simple_generation(prompt, num_samples=3)
                
                # 测试代码
                tested_results = self.test_code_solutions(problem, generation_results)
                
                # 保存详细结果
                problem_result = {
                    'task_id': task_id,
                    'problem_text': problem['text'],
                    'expected_function_name': expected_func_name,
                    'test_cases': problem['test_list'],
                    'generated_solutions': []
                }
                
                for i, result in enumerate(tested_results):
                    problem_result['generated_solutions'].append({
                        'solution_id': i + 1,
                        'code': result['code'],
                        'passed_tests': result['passed_tests'],
                        'total_tests': result['total_tests'],
                        'pass_rate': result['pass_rate'],
                        'fully_passed': result['fully_passed'],
                        'errors': result.get('test_errors', [])
                    })
                
                all_results.append(problem_result)
                
                success_count = sum(1 for r in tested_results if r['fully_passed'])
                print(f"  问题 {task_id} 完成，成功解决方案：{success_count}/{len(tested_results)}")
                
            except Exception as e:
                print(f"处理问题 {task_id} 时出错: {e}")
                continue
        
        # 保存结果
        results = {
            'experiment_config': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'collect_problems': collect_problems,
                'improvements': ['better_prompts', 'function_name_extraction', 'code_post_processing'],
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': all_results,
            'summary': self.generate_summary(all_results)
        }
        
        with open(output_path / "btp_improved_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n改进实验完成！结果保存到: {output_path}")
        self.print_summary(results['summary'])
        
        return results
    
    def generate_summary(self, all_results):
        """生成实验总结"""
        total_problems = len(all_results)
        total_solutions = sum(len(r['generated_solutions']) for r in all_results)
        successful_solutions = sum(
            sum(1 for s in r['generated_solutions'] if s['fully_passed']) 
            for r in all_results
        )
        
        problems_with_solution = sum(
            1 for r in all_results 
            if any(s['fully_passed'] for s in r['generated_solutions'])
        )
        
        return {
            'total_problems': total_problems,
            'total_solutions': total_solutions,
            'successful_solutions': successful_solutions,
            'solution_success_rate': successful_solutions / total_solutions if total_solutions > 0 else 0,
            'problems_solved': problems_with_solution,
            'problem_success_rate': problems_with_solution / total_problems if total_problems > 0 else 0
        }
    
    def print_summary(self, summary):
        """打印实验总结"""
        print(f"\n{'='*50}")
        print("实验总结")
        print(f"{'='*50}")
        print(f"总问题数: {summary['total_problems']}")
        print(f"总解决方案数: {summary['total_solutions']}")
        print(f"成功解决方案数: {summary['successful_solutions']}")
        print(f"解决方案成功率: {summary['solution_success_rate']:.2%}")
        print(f"问题解决数: {summary['problems_solved']}")
        print(f"问题成功率: {summary['problem_success_rate']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="改进的BTP实验")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--output_dir", type=str, default="experiments/results/btp_improved")
    parser.add_argument("--collect_problems", type=int, default=3)
    
    args = parser.parse_args()
    
    experiment = BTPExperimentImproved(args.model_name, args.dataset)
    results = experiment.run_experiment(args.output_dir, args.collect_problems)


if __name__ == "__main__":
    main() 