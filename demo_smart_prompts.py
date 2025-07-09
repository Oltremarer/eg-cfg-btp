#!/usr/bin/env python3
"""
智能Prompt模板系统演示脚本
演示如何为不同模型自动生成适配的prompt格式

运行方式:
python demo_smart_prompts.py
"""

import sys
import os

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.prompt_templates import (
    get_model_prompt, 
    detect_model_info, 
    validate_model_compatibility,
    ModelFamily,
    ModelType
)

def print_separator(title: str):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def demo_model_detection():
    """演示模型检测功能"""
    print_separator("🔍 模型检测演示")
    
    test_models = [
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "meta-llama/CodeLlama-7b-Instruct-hf",
        "meta-llama/CodeLlama-70b-Instruct-hf",  # 特殊格式
        "bigcode/starcoder-15b",
        "WizardLM/WizardCoder-15B-V1.0",
        "Qwen/CodeQwen1.5-7B-Chat", 
        "gpt-3.5-turbo",
        "claude-3-sonnet",
        "unknown-model-name"
    ]
    
    for model in test_models:
        model_info = detect_model_info(model)
        special = f" (特殊格式: {model_info.special_format})" if model_info.special_format else ""
        size = f" - {model_info.size}" if model_info.size else ""
        
        print(f"📋 {model}")
        print(f"   家族: {model_info.family.value}")
        print(f"   类型: {model_info.type.value}{size}{special}")
        print()

def demo_prompt_generation():
    """演示prompt生成功能"""
    print_separator("🎯 Prompt生成演示")
    
    # 示例问题
    sample_problem = {
        'text': 'Write a function to remove the first and last occurrence of a given character from the string.',
        'test_list': [
            'assert remove_Occ("hello","l") == "heo"',
            'assert remove_Occ("abcda","a") == "bcd"',
            'assert remove_Occ("PHP","P") == "H"'
        ]
    }
    
    test_cases = [
        {
            "model": "deepseek-ai/deepseek-coder-1.3b-instruct",
            "description": "DeepSeek指令模型 (支持few-shot)"
        },
        {
            "model": "meta-llama/CodeLlama-7b-Instruct-hf", 
            "description": "CodeLLaMA指令模型"
        },
        {
            "model": "meta-llama/CodeLlama-70b-Instruct-hf",
            "description": "CodeLLaMA-70B (特殊Source格式)"
        },
        {
            "model": "bigcode/starcoder-15b",
            "description": "StarCoder基础模型"
        },
        {
            "model": "WizardLM/WizardCoder-15B-V1.0", 
            "description": "WizardCoder (Alpaca格式)"
        },
        {
            "model": "gpt-3.5-turbo",
            "description": "OpenAI模型 (Messages格式)"
        }
    ]
    
    for case in test_cases:
        print(f"🤖 {case['description']}")
        print(f"   模型: {case['model']}")
        
        try:
            prompt = get_model_prompt(
                model_name=case['model'],
                dataset="mbpp",
                problem=sample_problem
            )
            
            if isinstance(prompt, list):
                # OpenAI/Claude的messages格式
                print("   格式: Messages")
                for msg in prompt:
                    role = msg['role']
                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    print(f"   {role}: {content}")
            else:
                # 字符串格式
                preview = prompt[:150] + "..." if len(prompt) > 150 else prompt
                print(f"   Prompt预览: {preview}")
                
        except Exception as e:
            print(f"   ❌ 生成失败: {e}")
        
        print()

def demo_few_shot_examples():
    """演示few-shot examples功能"""
    print_separator("📚 Few-shot Examples演示")
    
    sample_problem = {
        'text': 'Write a function to find the maximum element in a list.',
        'test_list': ['assert max_element([1, 2, 3]) == 3']
    }
    
    # 准备few-shot examples
    examples = [
        {
            "problem": "Write a function to find the similar elements from the given two tuple lists.",
            "test_cases": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)"
            ],
            "solution": """def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)"""
        }
    ]
    
    print("🧪 DeepSeek模型 - 不使用examples:")
    prompt1 = get_model_prompt(
        model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
        dataset="mbpp",
        problem=sample_problem,
        use_examples=False
    )
    print(prompt1[:200] + "...")
    
    print("\n🎓 DeepSeek模型 - 使用few-shot examples:")
    prompt2 = get_model_prompt(
        model_name="deepseek-ai/deepseek-coder-6.7b-instruct", 
        dataset="mbpp",
        problem=sample_problem,
        use_examples=True,
        examples=examples
    )
    print(prompt2[:400] + "...")

def demo_compatibility_check():
    """演示兼容性检查"""
    print_separator("✅ 兼容性检查演示")
    
    test_cases = [
        ("deepseek-ai/deepseek-coder-1.3b-instruct", "mbpp"),
        ("bigcode/starcoder-15b", "humaneval"),
        ("bigcode/starcoder-15b", "mbpp"),
        ("unknown-model", "mbpp")
    ]
    
    for model, dataset in test_cases:
        print(f"🔍 检查: {model} + {dataset}")
        
        compatibility = validate_model_compatibility(model, dataset)
        
        print(f"   状态: {'✅ 兼容' if compatibility['supported'] else '❌ 不兼容'}")
        
        if compatibility['recommendations']:
            print("   💡 建议:")
            for rec in compatibility['recommendations']:
                print(f"      - {rec}")
        
        if compatibility['warnings']:
            print("   ⚠️  警告:")
            for warning in compatibility['warnings']:
                print(f"      - {warning}")
        
        print()

def demo_dataset_differences():
    """演示不同数据集的格式差异"""
    print_separator("📊 数据集格式差异演示")
    
    datasets = [
        {
            "name": "mbpp",
            "problem": {
                'text': 'Write a function to find the maximum element in a list.',
                'test_list': ['assert max_element([1, 2, 3]) == 3']
            }
        },
        {
            "name": "humaneval", 
            "problem": {
                'prompt': '''def max_element(lst):
    """
    Find the maximum element in a list.
    
    Args:
        lst: A list of numbers
        
    Returns:
        The maximum element
        
    Examples:
    >>> max_element([1, 2, 3])
    3
    >>> max_element([10, 5, 8])
    10
    """'''
            }
        },
        {
            "name": "apps",
            "problem": {
                'question': 'Find the maximum element in a given list of integers.',
                'test_list': [
                    'assert max_element([1, 2, 3]) == 3',
                    'assert max_element([10, 5, 8]) == 10'
                ]
            }
        }
    ]
    
    model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
    
    for dataset_info in datasets:
        dataset_name = dataset_info["name"]
        problem = dataset_info["problem"]
        
        print(f"📋 数据集: {dataset_name.upper()}")
        
        prompt = get_model_prompt(
            model_name=model_name,
            dataset=dataset_name,
            problem=problem
        )
        
        preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        print(f"   格式化结果: {preview}")
        print()

def main():
    """主演示函数"""
    print("🚀 智能Prompt模板系统演示")
    print("基于《代码生成模型Prompt模板权威指南》")
    
    try:
        demo_model_detection()
        demo_prompt_generation()
        demo_few_shot_examples()
        demo_compatibility_check()
        demo_dataset_differences()
        
        print_separator("🎉 演示完成")
        print("💡 提示: 查看 experiments/model_prompt_usage_guide.md 获取详细使用指南")
        print("📝 提示: 现有实验可以逐步迁移到新系统，无需一次性重写")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已正确安装依赖并设置Python路径")
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 