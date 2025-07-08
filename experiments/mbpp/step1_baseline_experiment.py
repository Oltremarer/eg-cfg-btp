#!/usr/bin/env python3
"""
MBPP数据集 - Step1: 基线模型性能评测
基于共享基础类的MBPP特定实现
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入共享模块
from experiments.shared.base_experiment import Step1BaselineExperiment
from experiments.shared.dataset_configs import get_dataset_config, get_prompt_template
from experiments.shared.common_utils import (
    load_mbpp_problems, safe_execute_code, format_experiment_summary
)

# 导入EG-CFG模块
from eg_cfg.model_utils import setup_device, load_model, load_tokenizer
from eg_cfg.code_generation_utils import generate_code_solutions
from eg_cfg.openai_utils import OpenAIClient
from eg_cfg.mbpp_utils import run_tests


class MBPPBaselineExperiment(Step1BaselineExperiment):
    """MBPP数据集的基线实验"""
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载MBPP数据集配置"""
        return get_dataset_config("mbpp")
    
    def load_dataset(self) -> Dict[str, Any]:
        """加载MBPP数据集"""
        return load_mbpp_problems()
    
    def format_prompt(self, problem: Dict[str, Any]) -> str:
        """格式化MBPP提示词"""
        prompt_template = get_prompt_template("mbpp")
        
        prompt = f"""Solve the following programming problem:

Problem: {problem['text']}

Test cases:
{chr(10).join(problem['test_list'])}

Provide a complete Python function:

```python
"""
        return prompt
    
    def generate_solutions(self, problem: Dict[str, Any], num_samples: int, 
                          temperature: float) -> List[str]:
        """生成解决方案"""
        if self.use_openai:
            return self.generate_solutions_openai(problem, num_samples, temperature)
        else:
            return self.generate_solutions_local(problem, num_samples, temperature)
    
    def generate_solutions_openai(self, problem: Dict[str, Any], num_samples: int, 
                                 temperature: float) -> List[str]:
        """使用OpenAI API生成解决方案"""
        prompt = self.format_prompt(problem)
        
        try:
            if not hasattr(self, 'openai_client'):
                self.openai_client = OpenAIClient(model=self.model_name)
            
            solutions = self.openai_client.generate_code(
                prompt=prompt,
                max_tokens=512,
                temperature=temperature,
                n=num_samples
            )
            return solutions
            
        except Exception as e:
            print(f"OpenAI生成失败 问题 {problem.get('task_id', 'unknown')}: {e}")
            return []
    
    def generate_solutions_local(self, problem: Dict[str, Any], num_samples: int, 
                                temperature: float) -> List[str]:
        """使用本地模型生成解决方案"""
        try:
            # 如果还没有加载模型，现在加载
            if not hasattr(self, 'model') or self.model is None:
                self.device = setup_device()
                self.model, self.tokenizer = load_model(self.model_name, self.device)
            
            prompt = self.format_prompt(problem)
            solutions = []
            
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
                    code_parts = generated_text.split("```python")
                    if len(code_parts) > 1:
                        code = code_parts[1].split("```")[0].strip()
                    else:
                        code = generated_text[len(prompt):].strip()
                elif "```" in generated_text:
                    code_parts = generated_text.split("```")
                    if len(code_parts) > 2:
                        code = code_parts[1].strip()
                    else:
                        code = generated_text[len(prompt):].strip()
                else:
                    code = generated_text[len(prompt):].strip()
                
                # 清理代码
                if code:
                    for end_marker in ["```", "<|endoftext|>", "<|EOT|>", "\n\n\n"]:
                        if end_marker in code:
                            code = code.split(end_marker)[0]
                    
                    code = code.strip()
                    if code:
                        solutions.append(code)
                        print(f"    生成的代码: {code[:50]}...")
            
            print(f"  成功生成 {len(solutions)} 个解决方案")
            return solutions
            
        except Exception as e:
            print(f"本地模型生成失败 问题 {problem.get('task_id', 'unknown')}: {e}")
            return []
    
    def test_solutions(self, problem: Dict[str, Any], solutions: List[str]) -> List[Dict[str, Any]]:
        """测试解决方案"""
        tested_solutions = []
        
        for i, solution in enumerate(solutions):
            try:
                # 使用EG-CFG的测试工具
                test_results = run_tests(solution, problem['test_list'])
                
                # 转换为标准格式
                passed_tests = sum(1 for r in test_results.values() if r['result'])
                total_tests = len(test_results)
                
                solution_result = {
                    'solution_id': i,
                    'solution': solution,
                    'passed_tests': passed_tests,
                    'total_tests': total_tests,
                    'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                    'fully_passed': (passed_tests == total_tests and total_tests > 0),
                    'test_results': test_results
                }
                
                tested_solutions.append(solution_result)
                
                # 打印调试信息
                print(f"  解决方案 {i}: {passed_tests}/{total_tests} 测试通过")
                if not solution_result['fully_passed']:
                    print(f"    代码: {solution[:100]}...")
                
            except Exception as e:
                print(f"测试解决方案 {i} 失败: {e}")
                solution_result = {
                    'solution_id': i,
                    'solution': solution,
                    'passed_tests': 0,
                    'total_tests': len(problem['test_list']),
                    'pass_rate': 0.0,
                    'fully_passed': False,
                    'test_results': {},
                    'error': str(e)
                }
                tested_solutions.append(solution_result)
        
        return tested_solutions


def main():
    parser = argparse.ArgumentParser(description="MBPP Baseline Experiment")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--output_dir", type=str, default="experiments/mbpp/results", 
                       help="输出目录")
    parser.add_argument("--num_samples", type=int, default=10, 
                       help="每个问题的采样数")
    parser.add_argument("--temperature", type=float, default=0.8, 
                       help="采样温度")
    parser.add_argument("--max_problems", type=int, default=None, 
                       help="最大问题数量")
    parser.add_argument("--use_openai", action="store_true", 
                       help="使用OpenAI API")
    
    args = parser.parse_args()
    
    results = {}
    pass_at_k = {}
    error_info = None
    
    try:
        print(f"开始MBPP基线实验: {args.model_name}")
        
        # 创建实验实例
        experiment = MBPPBaselineExperiment(
            dataset_name="mbpp",
            model_name=args.model_name,
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
        
    except Exception as e:
        error_info = str(e)
        print(f"实验失败: {e}")
        print("保存部分结果...")
        import traceback
        traceback.print_exc()
    
    # 保存结果
    try:
        final_results = {
            'experiment_config': {
                'model_name': args.model_name,
                'dataset': 'mbpp',
                'num_samples': args.num_samples,
                'temperature': args.temperature,
                'max_problems': args.max_problems,
                'use_openai': args.use_openai
            },
            'results': results,
            'metrics': pass_at_k,
            'error_info': error_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # 使用基础类的保存方法
        filepath = experiment.save_results(final_results, "baseline")
        
        # 打印结果摘要
        summary = format_experiment_summary(final_results, experiment.get_experiment_config())
        print(summary)
        
        return 0 if error_info is None else 1
        
    except Exception as save_error:
        print(f"保存结果失败: {save_error}")
        return 1


if __name__ == "__main__":
    exit(main()) 