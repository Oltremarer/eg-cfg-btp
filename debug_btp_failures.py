#!/usr/bin/env python3
"""
调试BTP实验失败原因
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests
from eg_cfg.model_utils import setup_device, load_model, load_tokenizer

def debug_single_problem():
    """调试单个问题的处理过程"""
    
    print("=== BTP失败调试分析 ===")
    
    # 加载数据
    problems = load_mbpp_problems()
    first_problem = list(problems.items())[0]  # 取第一个问题
    task_id, problem = first_problem
    
    print(f"调试问题 {task_id}:")
    print(f"问题描述: {problem['text']}")
    print(f"测试用例: {problem['test_list']}")
    
    # 模拟生成的代码（从DeepSeek可能生成的内容）
    sample_generated_codes = [
        "def function(x):\n    return x + 1",  # 简单错误代码
        "def solve():\n    pass",  # 空函数
        "print('hello world')",  # 不是函数
        "def test_function(input_val):\n    return input_val * 2",  # 函数名错误
        "def " + problem['text'].split()[0] + "():\n    return None"  # 无参数函数
    ]
    
    print(f"\n=== 模拟测试 {len(sample_generated_codes)} 个生成的代码 ===")
    
    for i, code in enumerate(sample_generated_codes):
        print(f"\n--- 代码 {i+1} ---")
        print(f"生成的代码:")
        print(code)
        
        try:
            # 测试代码
            test_results = run_tests(code, problem['test_list'])
            
            passed_tests = sum(1 for r in test_results.values() if r.get('result', False))
            total_tests = len(test_results)
            
            print(f"测试结果: {passed_tests}/{total_tests} 通过")
            
            for test_case, result in test_results.items():
                status = "✅" if result['result'] else "❌"
                print(f"  {status} {test_case}")
                if not result['result'] and result.get('error'):
                    print(f"    错误: {result['error']}")
                    
        except Exception as e:
            print(f"❌ 测试过程出错: {e}")

def analyze_common_failures():
    """分析常见的失败模式"""
    
    print(f"\n=== 常见失败模式分析 ===")
    
    common_issues = [
        "1. 函数名不匹配 - 生成的函数名与问题要求不符",
        "2. 参数错误 - 参数数量或名称不正确", 
        "3. 返回值错误 - 返回值类型或格式不对",
        "4. 逻辑错误 - 算法实现有误",
        "5. 语法错误 - 生成的代码有语法问题",
        "6. 导入缺失 - 需要的库没有导入",
        "7. 空函数 - 只有函数框架没有实现"
    ]
    
    for issue in common_issues:
        print(f"  {issue}")

def suggest_improvements():
    """建议改进方案"""
    
    print(f"\n=== 改进建议 ===")
    
    suggestions = [
        "1. 使用更好的提示词模板",
        "2. 增加few-shot示例",
        "3. 后处理生成的代码（修复常见错误）",
        "4. 调整beam search参数",
        "5. 使用更大的模型",
        "6. 增加代码验证步骤"
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

if __name__ == "__main__":
    debug_single_problem()
    analyze_common_failures() 
    suggest_improvements() 