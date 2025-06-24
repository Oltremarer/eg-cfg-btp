#!/usr/bin/env python3
"""
Experiment 2 English Version: Using Standard English Prompts
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


class BTPExperimentEnglish:
    """BTP Experiment with Standard English Prompts"""
    
    def __init__(self, model_name: str, dataset: str):
        self.model_name = model_name
        self.dataset = dataset
        
        print(f"Loading model: {model_name}")
        self.device = setup_device()
        
        if "cuda" in str(self.device):
            print("CUDA detected but forcing CPU for stability")
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
        """Extract function name from test cases"""
        for test in test_list:
            if 'assert' in test:
                # Match assert function_name(...) pattern
                match = re.search(r'assert\s+(\w+)\s*\(', test)
                if match:
                    return match.group(1)
        return "solution"  # Default function name
    
    def create_standard_english_prompt(self, problem):
        """Create standard English prompt based on DeepSeek official template"""
        # Extract function name
        func_name = self.extract_function_name_from_tests(problem['test_list'])
        
        # Use improved prompt format
        prompt = f"""You are an AI programming assistant. Generate a Python function to solve the following problem.

Problem: {problem['text']}

Test Cases:
{chr(10).join(problem['test_list'])}

Write a complete Python function named '{func_name}' that passes all test cases:

```python"""
        
        return prompt, func_name
    
    def simple_generation(self, prompt, num_samples=3, max_tokens=200):
        """Simple code generation"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract expected function name from prompt
        if "def " in prompt:
            expected_func_name = prompt.split("def ")[-1].split("(")[0].strip()
        else:
            expected_func_name = "solution"
        
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
            
            # Extract Python code block
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].strip()
            
            # Ensure complete function
            code = self.ensure_complete_function(code, expected_func_name)
            
            sequence_length = outputs[0].shape[0] - inputs['input_ids'].shape[1]
            log_prob = -sequence_length * 0.1
            
            results.append({
                'code': code,
                'log_prob': log_prob,
                'sequence_length': sequence_length,
                'sample_rank': i + 1
            })
            
            print(f"    Generated sample {i+1}: {code[:100]}...")
        
        return results
    
    def ensure_complete_function(self, code, expected_func_name):
        """Ensure function is complete"""
        code = code.strip()
        
        # If doesn't start with def, try to fix
        if not code.startswith('def '):
            # Look for def line
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    code = '\n'.join(lines[i:])
                    break
            else:
                # If no def found, create a simple template
                if code and not code.startswith('def '):
                    # Try to create a meaningful function signature
                    if 'string' in code.lower() or 'str' in code.lower():
                        code = f"def {expected_func_name}(string, char):\n    {code}"
                    elif 'matrix' in code.lower():
                        code = f"def {expected_func_name}(matrix):\n    {code}"
                    elif 'list' in code.lower() or 'words' in code.lower():
                        code = f"def {expected_func_name}(words):\n    {code}"
                    else:
                        code = f"def {expected_func_name}():\n    {code}"
        
        # Basic indentation fix
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
        """Test code solutions"""
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
    
    def run_experiment(self, output_dir, collect_problems=3, num_samples=3, max_tokens=200):
        """Run English prompt experiment"""
        print("Starting BTP experiment with standard English prompts...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        problems_list = list(self.problems.items())[:collect_problems]
        all_results = []
        
        for task_id, problem in tqdm(problems_list, desc="Processing problems"):
            try:
                print(f"\nProcessing problem {task_id}: {problem['text'][:50]}...")
                
                # Create standard English prompt
                prompt, expected_func_name = self.create_standard_english_prompt(problem)
                print(f"Expected function name: {expected_func_name}")
                
                # Generate code
                generation_results = self.simple_generation(prompt, num_samples=num_samples, max_tokens=max_tokens)
                
                # Test code
                tested_results = self.test_code_solutions(problem, generation_results)
                
                # Save detailed results
                problem_result = {
                    'task_id': task_id,
                    'problem_text': problem['text'],
                    'expected_function_name': expected_func_name,
                    'test_cases': problem['test_list'],
                    'prompt_template': 'deepseek_official_english',
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
                print(f"  Problem {task_id} completed, successful solutions: {success_count}/{len(tested_results)}")
                
            except Exception as e:
                print(f"Error processing problem {task_id}: {e}")
                continue
        
        # Save results
        results = {
            'experiment_config': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'collect_problems': collect_problems,
                'prompt_type': 'deepseek_official_english',
                'improvements': ['standard_english_prompts', 'deepseek_template', 'function_name_extraction'],
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': all_results,
            'summary': self.generate_summary(all_results)
        }
        
        with open(output_path / "btp_english_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nEnglish prompt experiment completed! Results saved to: {output_path}")
        self.print_summary(results['summary'])
        
        return results
    
    def generate_summary(self, all_results):
        """Generate experiment summary"""
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
        """Print experiment summary"""
        print(f"\n{'='*50}")
        print("Experiment Summary")
        print(f"{'='*50}")
        print(f"Total problems: {summary['total_problems']}")
        print(f"Total solutions: {summary['total_solutions']}")
        print(f"Successful solutions: {summary['successful_solutions']}")
        print(f"Solution success rate: {summary['solution_success_rate']:.2%}")
        print(f"Problems solved: {summary['problems_solved']}")
        print(f"Problem success rate: {summary['problem_success_rate']:.2%}")


def main():
    # 预定义的模型选项
    AVAILABLE_MODELS = {
        "smollm-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "smollm-360m": "HuggingFaceTB/SmolLM2-360M-Instruct", 
        "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "deepseek-v2-lite": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "codegemma-2b": "google/codegemma-2b-it",
        "codegemma-7b": "google/codegemma-7b-it"
    }
    
    parser = argparse.ArgumentParser(description="BTP Experiment - Standard English Prompts")
    parser.add_argument("--model_name", type=str, 
                       help=f"Model name, can use shortcuts: {list(AVAILABLE_MODELS.keys())} or full HuggingFace model path")
    parser.add_argument("--model_preset", type=str, choices=list(AVAILABLE_MODELS.keys()),
                       default="smollm-135m", 
                       help="Preset model options (recommended for CPU: smollm-135m)")
    parser.add_argument("--dataset", type=str, default="mbpp", choices=["mbpp", "humaneval"])
    parser.add_argument("--output_dir", type=str, default="experiments/results/btp_english")
    parser.add_argument("--collect_problems", type=int, default=3,
                       help="Number of problems to process (CPU recommended: 3-5)")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of code samples per problem (CPU recommended: 2-3)")
    parser.add_argument("--max_tokens", type=int, default=200,
                       help="Maximum tokens to generate per sample")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU usage")
    
    args = parser.parse_args()
    
    # 确定使用的模型
    if args.model_name:
        if args.model_name in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[args.model_name]
            print(f"Using preset model: {args.model_name} -> {model_name}")
        else:
            model_name = args.model_name
            print(f"Using custom model: {model_name}")
    else:
        model_name = AVAILABLE_MODELS[args.model_preset]
        print(f"Using default model: {args.model_preset} -> {model_name}")
    
    print(f"Experiment Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Problems: {args.collect_problems}")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Force CPU: {args.force_cpu}")
    
    # 创建实验实例
    experiment = BTPExperimentEnglish(model_name, args.dataset)
    
    # 如果强制使用CPU
    if args.force_cpu:
        experiment.device = torch.device("cpu")
        experiment.model = experiment.model.to(experiment.device)
        print("Forced to CPU mode")
    
    # 运行实验
    results = experiment.run_experiment(args.output_dir, args.collect_problems, args.num_samples, args.max_tokens)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Experiment completed!")


if __name__ == "__main__":
    main() 