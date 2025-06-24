#!/usr/bin/env python3
"""
测试OpenAI版本的BTP实验
这是一个简化的测试脚本，用于验证OpenAI BTP实验是否能正常工作
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_openai_connection():
    """测试OpenAI连接"""
    print("测试OpenAI连接...")
    
    try:
        from eg_cfg.openai_utils import OpenAIClient
        
        # 检查API密钥
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("❌ 未设置OPENAI_API_KEY环境变量")
            print("请设置: export OPENAI_API_KEY='your-api-key'")
            return False
        
        # 创建客户端
        client = OpenAIClient(api_key=api_key, model='gpt-3.5-turbo')
        
        # 测试简单查询
        test_prompt = "请写一个Python函数计算两个数的和："
        result = client.generate_code(test_prompt, max_tokens=100, n=1)
        
        if result:
            print("✅ OpenAI连接测试成功")
            print(f"生成的代码示例: {result[0][:100]}...")
            return True
        else:
            print("❌ OpenAI API调用失败")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI连接测试失败: {e}")
        return False


def test_mbpp_loading():
    """测试MBPP数据加载"""
    print("\n测试MBPP数据加载...")
    
    try:
        from eg_cfg.mbpp_utils import load_mbpp_problems
        
        problems = load_mbpp_problems()
        print(f"✅ 成功加载 {len(problems)} 个MBPP问题")
        
        # 显示第一个问题示例
        first_problem = next(iter(problems.values()))
        print(f"第一个问题示例: {first_problem['text'][:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ MBPP数据加载失败: {e}")
        return False


def run_simple_btp_test():
    """运行简单的BTP测试"""
    print("\n运行简单BTP测试...")
    
    try:
        # 确保API密钥可用
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("❌ 需要设置OPENAI_API_KEY环境变量")
            return False
        
        # 这里我们创建一个简化的BTP测试
        from eg_cfg.openai_utils import OpenAIClient
        from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests
        
        print("加载数据和模型...")
        problems = load_mbpp_problems()
        client = OpenAIClient(api_key=api_key, model='gpt-3.5-turbo')
        
        # 选择第一个问题进行测试
        first_task_id = list(problems.keys())[0]
        first_problem = problems[first_task_id]
        
        print(f"测试问题 {first_task_id}: {first_problem['text'][:100]}...")
        
        # 格式化提示词
        prompt = f"""请解决以下编程问题：

问题描述：
{first_problem['text']}

要求：
1. 请提供一个完整的Python函数实现
2. 函数应该能够通过所有测试用例
3. 代码应该简洁、高效且正确

请在代码块中提供解决方案：

```python
"""
        
        # 生成多个候选解决方案（模拟beam search）
        print("生成候选解决方案...")
        solutions = client.generate_code(
            prompt=prompt,
            max_tokens=512,
            temperature=0.8,
            n=3  # 生成3个候选
        )
        
        if not solutions:
            print("❌ 未能生成解决方案")
            return False
        
        print(f"生成了 {len(solutions)} 个候选解决方案")
        
        # 测试每个解决方案
        best_solution = None
        best_score = 0
        
        for i, code in enumerate(solutions):
            print(f"\n测试解决方案 {i+1}:")
            print(f"代码: {code[:200]}...")
            
            try:
                test_results = run_tests(code, first_problem['test_list'])
                passed_tests = sum(1 for r in test_results.values() if r.get('result', False))
                total_tests = len(test_results)
                pass_rate = passed_tests / total_tests if total_tests > 0 else 0
                
                print(f"测试结果: {passed_tests}/{total_tests} ({pass_rate:.1%})")
                
                if pass_rate > best_score:
                    best_score = pass_rate
                    best_solution = code
                
            except Exception as e:
                print(f"测试失败: {e}")
        
        print(f"\n最佳解决方案成功率: {best_score:.1%}")
        if best_score > 0:
            print("✅ 简单BTP测试成功")
            return True
        else:
            print("⚠️  BTP测试完成，但没有解决方案通过所有测试")
            return False
            
    except Exception as e:
        print(f"❌ BTP测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("OpenAI BTP实验测试")
    print("=" * 50)
    
    # 运行各项测试
    tests = [
        ("OpenAI连接", test_openai_connection),
        ("MBPP数据加载", test_mbpp_loading),
        ("简单BTP测试", run_simple_btp_test)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed_tests += 1
        else:
            print(f"⚠️  {test_name} 失败，后续测试可能受到影响")
    
    print("\n" + "=" * 50)
    print(f"测试总结: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！可以运行完整的BTP实验了")
        print("\n下一步:")
        print("1. 运行: python experiments/run_btp_openai_experiment.py")
        print("2. 或者使用: python experiments/step2_btp_experiment_openai.py --problems 10")
    else:
        print("❌ 部分测试失败，请检查配置")
        if passed_tests == 0:
            print("\n常见问题:")
            print("1. 确保设置了OPENAI_API_KEY环境变量")
            print("2. 确保API密钥有效且有足够余额")
            print("3. 确保网络连接正常")


if __name__ == "__main__":
    main() 