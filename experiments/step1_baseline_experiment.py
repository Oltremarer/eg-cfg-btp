#!/usr/bin/env python3
"""
实验一：基线模型性能评测 (Baseline Performance)
基于EG-CFG框架修改，支持标准的采样-过滤流程
支持OpenAI API
"""

import os
import sys
import json
import torch
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

# Add parent directory to path to import EG-CFG modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from eg_cfg.mbpp_utils import load_mbpp_problems, load_humaneval_problems, run_tests
from eg_cfg.model_utils import setup_device, load_model, load_tokenizer
from eg_cfg.code_generation_utils import generate_code_solutions, raw_outputs_to_new_code
from eg_cfg.openai_utils import OpenAIClient, openai_simple_query
from eg_cfg.consts import *
from datasets import load_dataset


class BaselineExperiment:
    def __init__(self, model_name: str, dataset: str, device: str = "cuda", use_openai: bool = False):
        self.model_name = model_name
        self.dataset = dataset
        self.device = device
        self.use_openai = use_openai
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        if use_openai:
            print("Using OpenAI API")
            self.openai_client = OpenAIClient(model=model_name)
            self.model = None
            self.tokenizer = None
            self.use_local_model = False
        elif "inference_endpoint" in model_name.lower():
            self.model = None
            self.tokenizer = load_tokenizer(model_name)
            self.use_local_model = False
        else:
            self.device = setup_device()
            self.model, self.tokenizer = load_model(model_name, self.device)
            self.use_local_model = True
        
        # Load dataset
        print(f"Loading dataset: {dataset}")
        if dataset == "mbpp":
            self.problems = load_mbpp_problems()
        elif dataset == "humaneval":
            self.problems = load_humaneval_problems()
        elif dataset == "apps":
            self.problems = self.load_apps_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        print(f"Loaded {len(self.problems)} problems")
    
    def load_apps_dataset(self, split="test", difficulty_filter=None):
        """Load APPS dataset"""
        dataset = load_dataset("codeparrot/apps", split=split)
        problems = {}
        
        for idx, problem in enumerate(dataset):
            if difficulty_filter and problem.get('difficulty') not in difficulty_filter:
                continue
                
            # Format for consistency with MBPP/HumanEval
            problems[idx] = {
                'task_id': idx,
                'question': problem['question'],
                'solutions': problem.get('solutions', []),
                'input_output': problem.get('input_output', {}),
                'difficulty': problem.get('difficulty', 'unknown'),
                # Create test cases from input_output
                'test_list': self.extract_apps_test_cases(problem.get('input_output', {}))
            }
        
        return problems
    
    def extract_apps_test_cases(self, input_output):
        """Extract test cases from APPS input_output format"""
        test_cases = []
        if isinstance(input_output, dict) and 'inputs' in input_output and 'outputs' in input_output:
            inputs = input_output['inputs']
            outputs = input_output['outputs']
            
            if isinstance(inputs, list) and isinstance(outputs, list):
                for inp, out in zip(inputs, outputs):
                    # Create assertion-style test case
                    test_case = f"assert solution({repr(inp)}) == {repr(out)}"
                    test_cases.append(test_case)
        
        return test_cases if test_cases else ["# No test cases available"]
    
    def generate_solutions_openai(self, problem, num_samples=10, temperature=0.8):
        """使用OpenAI API生成解决方案"""
        # 构造提示
        if self.dataset == "mbpp":
            prompt = f"""
请解决以下编程问题：

问题描述：
{problem['text']}

测试用例：
{chr(10).join(problem['test_list'])}

请编写一个Python函数来解决这个问题。只返回函数代码，不要包含额外的解释。
"""
        elif self.dataset == "humaneval":
            prompt = f"""
请完成以下Python函数：

{problem['prompt']}

请只返回完整的函数实现。
"""
        else:
            # APPS dataset
            prompt = f"""
请解决以下编程问题：

{problem['question']}

测试用例：
{chr(10).join(problem['test_list'])}

请编写一个Python函数来解决这个问题。
"""
        
        try:
            solutions = self.openai_client.generate_code(
                prompt=prompt,
                max_tokens=512,
                temperature=temperature,
                n=num_samples
            )
            return solutions
        except Exception as e:
            print(f"Error generating solutions for problem {problem.get('task_id', 'unknown')}: {e}")
            return []
    
    def run_experiment(self, num_samples=10, temperature=0.8, max_problems=None):
        """运行基线实验"""
        print(f"Running baseline experiment with {num_samples} samples per problem")
        
        # 选择要测试的问题
        problem_ids = list(self.problems.keys())
        if max_problems:
            problem_ids = problem_ids[:max_problems]
        
        results = {}
        
        for problem_id in tqdm(problem_ids, desc="Processing problems"):
            problem = self.problems[problem_id]
            print(f"\nProcessing problem {problem_id}")
            
            # 生成解决方案
            if self.use_openai:
                solutions = self.generate_solutions_openai(problem, num_samples, temperature)
            else:
                # 使用现有的本地模型或推理端点逻辑
                solutions = self.generate_solutions_local(problem, num_samples, temperature)
            
            if not solutions:
                print(f"No solutions generated for problem {problem_id}")
                continue
            
            # 测试解决方案
            problem_results = []
            for i, solution in enumerate(solutions):
                try:
                    # 运行测试
                    test_results = run_tests(solution, problem['test_list'])
                    
                    # 添加详细调试信息
                    print(f"  Solution {i}:")
                    print(f"    Code: {solution[:100]}...")
                    print(f"    Full Code: {repr(solution)}")  # 显示完整代码和隐藏字符
                    
                    for test_case, result in test_results.items():
                        print(f"    测试: {test_case}")
                        print(f"    结果: {'✅' if result['result'] else '❌'}")
                        if result.get('error'):
                            print(f"    错误: {result['error']}")
                        print(f"    执行时间: {result['time']:.4f}s")
                        if not result['result']:
                            print(f"    ❌ FAILED: {test_case}")
                            if result.get('error'):
                                print(f"        Error: {result['error']}")
                    
                    # run_tests 返回的字典中每个值都有 'result' 键而不是 'passed'
                    passed_tests = sum(1 for test_case, result in test_results.items() if result['result'])
                    total_tests = len(test_results)
                    pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
                    fully_passed = pass_rate == 1.0
                    
                    if fully_passed:
                        print(f"    ✅ All tests passed!")
                    
                    problem_results.append({
                        'solution_id': i,
                        'solution': solution,
                        'passed_tests': passed_tests,
                        'total_tests': total_tests,
                        'pass_rate': pass_rate,
                        'fully_passed': fully_passed,
                        'test_results': test_results
                    })
                    
                except Exception as e:
                    print(f"Error testing solution {i}: {e}")
                    problem_results.append({
                        'solution_id': i,
                        'solution': solution,
                        'passed_tests': 0,
                        'total_tests': len(problem['test_list']),
                        'pass_rate': 0.0,
                        'fully_passed': False,
                        'error': str(e)
                    })
            
            results[problem_id] = {
                'problem': problem,
                'solutions': problem_results,
                'best_pass_rate': max(r['pass_rate'] for r in problem_results) if problem_results else 0.0,
                'any_fully_passed': any(r['fully_passed'] for r in problem_results)
            }
        
        return results
    
    def generate_solutions_local(self, problem, num_samples, temperature):
        """使用本地模型生成解决方案"""
        if not self.use_local_model:
            print("Local model not loaded")
            return []
        
        # 构造提示
        if self.dataset == "mbpp":
            prompt = f"""
问题描述：
{problem['text']}

测试用例：
{chr(10).join(problem['test_list'])}

请编写一个Python函数来解决这个问题：

```python
"""
        elif self.dataset == "humaneval":
            prompt = f"""{problem['prompt']}"""
        else:
            # APPS dataset
            prompt = f"""
问题：
{problem['question']}

测试用例：
{chr(10).join(problem['test_list'])}

请编写一个Python函数来解决这个问题：

```python
"""
        
        solutions = []
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self, 'device') and self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成多个解决方案
            for i in range(num_samples):
                print(f"  生成解决方案 {i+1}/{num_samples}...")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                # 解码输出
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取代码部分
                if "```python" in generated_text:
                    # 提取```python和```之间的代码
                    code_parts = generated_text.split("```python")
                    if len(code_parts) > 1:
                        code = code_parts[1].split("```")[0].strip()
                    else:
                        code = generated_text[len(prompt):].strip()
                elif "```" in generated_text:
                    # 提取```之后的代码
                    code_parts = generated_text.split("```")
                    if len(code_parts) > 2:
                        code = code_parts[1].strip()
                    else:
                        code = generated_text[len(prompt):].strip()
                else:
                    # 去掉原始提示，保留生成的部分
                    code = generated_text[len(prompt):].strip()
                
                # 清理代码
                if code:
                    # 去掉可能的结束标记
                    for end_marker in ["```", "<|endoftext|>", "<|EOT|>", "\n\n\n"]:
                        if end_marker in code:
                            code = code.split(end_marker)[0]
                    
                    code = code.strip()
                    if code:
                        solutions.append(code)
                        print(f"    生成的代码: {code[:50]}...")
        
        except Exception as e:
            print(f"    生成过程中出错: {e}")
        
        print(f"  成功生成 {len(solutions)} 个解决方案")
        return solutions
    
    def calculate_pass_at_k(self, all_results, k_values=[1, 5, 10]):
        """Calculate Pass@k metrics"""
        pass_at_k = {}
        
        for k in k_values:
            passed_problems = 0
            total_problems = len(all_results)
            
            for problem_id, problem_results in all_results.items():
                solutions = problem_results['solutions']
                if len(solutions) >= k:
                    # Check if any of the top k solutions fully passed
                    top_k_results = sorted(solutions, key=lambda x: x['pass_rate'], reverse=True)[:k]
                    if any(r['fully_passed'] for r in top_k_results):
                        passed_problems += 1
            
            pass_at_k[f'pass@{k}'] = passed_problems / total_problems if total_problems > 0 else 0.0
        
        return pass_at_k


def main():
    parser = argparse.ArgumentParser(description="Baseline Performance Experiment")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, choices=["mbpp", "humaneval", "apps"], 
                       required=True, help="Dataset to use")
    parser.add_argument("--output_dir", type=str, default="experiments/results/baseline", 
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10, 
                       help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=0.8, 
                       help="Sampling temperature")
    parser.add_argument("--max_problems", type=int, default=None, 
                       help="Maximum number of problems to evaluate")
    parser.add_argument("--use_openai", action="store_true", 
                       help="Use OpenAI API instead of local model")
    
    args = parser.parse_args()
    
    print(f"Starting baseline experiment with {args.model_name} on {args.dataset}")
    
    # 创建实验实例
    experiment = BaselineExperiment(
        model_name=args.model_name,
        dataset=args.dataset,
        use_openai=args.use_openai
    )
    
    # 运行实验
    results = experiment.run_experiment(
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_problems=args.max_problems
    )
    
    # 计算指标
    pass_at_k = experiment.calculate_pass_at_k(results)
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"baseline_results_{timestamp}.json"
    
    final_results = {
        'experiment_config': {
            'model_name': args.model_name,
            'dataset': args.dataset,
            'num_samples': args.num_samples,
            'temperature': args.temperature,
            'max_problems': args.max_problems,
            'use_openai': args.use_openai
        },
        'results': results,
        'metrics': pass_at_k,
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # 打印结果
    print(f"\n{'='*50}")
    print("EXPERIMENT RESULTS")
    print(f"{'='*50}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Problems tested: {len(results)}")
    print(f"Use OpenAI: {args.use_openai}")
    print(f"\nPass@k metrics:")
    for metric, value in pass_at_k.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()