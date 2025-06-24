#!/usr/bin/env python3
"""
调试OpenAI BTP实验中的问题
主要解决函数名不匹配导致的测试失败问题
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_task_11():
    """调试task_id=11的具体问题"""
    print("调试 Task 11 - 移除字符串中第一次和最后一次出现的字符")
    
    # 正确的测试用例
    test_cases = [
        'assert remove_Occ("hello","l") == "heo"',
        'assert remove_Occ("abcda","a") == "bcd"', 
        'assert remove_Occ("PHP","P") == "H"'
    ]
    
    print("测试用例:")
    for i, test in enumerate(test_cases, 1):
        print(f"  {i}. {test}")
    
    # 分析期望的行为
    print("\n分析期望行为:")
    print('  remove_Occ("hello","l") -> "heo"')
    print('    - 原始: "hello"')
    print('    - 移除第一个 "l" (位置1): "helo"') 
    print('    - 移除最后一个 "l" (位置2): "heo"')
    print()
    print('  remove_Occ("abcda","a") -> "bcd"')
    print('    - 原始: "abcda"')
    print('    - 移除第一个 "a" (位置0): "bcda"')
    print('    - 移除最后一个 "a" (位置3): "bcd"')
    
    # 正确的实现
    def remove_Occ(s, char):
        """正确的实现：移除第一次和最后一次出现的字符"""
        if char not in s:
            return s
        
        # 找到第一次和最后一次出现的位置
        first_index = s.find(char)
        last_index = s.rfind(char)
        
        # 如果只有一次出现，移除一次
        if first_index == last_index:
            return s[:first_index] + s[first_index + 1:]
        
        # 先移除后面的，再移除前面的（避免索引变化）
        result = s[:last_index] + s[last_index + 1:]
        first_index_in_result = result.find(char)
        result = result[:first_index_in_result] + result[first_index_in_result + 1:]
        
        return result
    
    print("\n测试正确的实现:")
    for test in test_cases:
        # 解析测试用例
        test_parts = test.split('==')
        left_part = test_parts[0].strip().replace('assert ', '')
        expected = test_parts[1].strip().strip('"')
        
        # 解析函数调用
        import re
        match = re.match(r'remove_Occ\("([^"]+)","([^"]+)"\)', left_part)
        if match:
            s, char = match.groups()
            result = remove_Occ(s, char)
            passed = result == expected
            print(f"  {left_part} -> '{result}' (期望: '{expected}') {'✅' if passed else '❌'}")
    
    return remove_Occ


def test_improved_openai_btp():
    """测试改进后的OpenAI BTP实验"""
    print("\n" + "="*60)
    print("测试改进后的OpenAI BTP实验")
    print("="*60)
    
    try:
        # 确保API密钥可用
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("❌ 需要设置OPENAI_API_KEY环境变量")
            return False
        
        from eg_cfg.openai_utils import OpenAIClient
        from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests
        
        print("加载数据和模型...")
        problems = load_mbpp_problems()
        client = OpenAIClient(api_key=api_key, model='gpt-3.5-turbo')
        
        # 获取task_id=11的问题
        problem = problems[11]
        
        print(f"问题描述: {problem['text']}")
        print(f"测试用例: {problem['test_list']}")
        
        # 改进的提示词，明确指定函数名
        prompt = f"""请解决以下编程问题：

问题描述：
{problem['text']}

要求：
1. 函数名必须是 remove_Occ
2. 函数应该移除字符串中指定字符的第一次和最后一次出现
3. 如果字符只出现一次，则移除那一次
4. 如果字符不存在，返回原字符串

测试用例：
{chr(10).join(problem['test_list'])}

请在代码块中提供解决方案：

```python
def remove_Occ(s, char):
"""
        
        print("生成解决方案...")
        solutions = client.generate_code(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            n=3
        )
        
        if not solutions:
            print("❌ 未能生成解决方案")
            return False
        
        print(f"生成了 {len(solutions)} 个候选解决方案")
        
        # 测试每个解决方案
        best_solution = None
        best_score = 0
        
        for i, code in enumerate(solutions):
            print(f"\n--- 解决方案 {i+1} ---")
            print(f"代码:\n{code}")
            
            try:
                test_results = run_tests(code, problem['test_list'])
                passed_tests = sum(1 for r in test_results.values() if r.get('result', False))
                total_tests = len(test_results)
                pass_rate = passed_tests / total_tests if total_tests > 0 else 0
                
                print(f"测试结果: {passed_tests}/{total_tests} ({pass_rate:.1%})")
                
                # 显示详细测试结果
                for test_case, result in test_results.items():
                    status = "✅" if result.get('result', False) else "❌"
                    print(f"  {status} {test_case}")
                    if not result.get('result', False) and result.get('error'):
                        print(f"     错误: {result['error']}")
                
                if pass_rate > best_score:
                    best_score = pass_rate
                    best_solution = code
                
            except Exception as e:
                print(f"测试失败: {e}")
        
        print(f"\n{'='*60}")
        print(f"最佳解决方案成功率: {best_score:.1%}")
        if best_score > 0:
            print("✅ 改进的BTP测试成功")
            return True
        else:
            print("⚠️  仍然需要进一步调试")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("OpenAI BTP实验调试工具")
    print("="*50)
    
    # 步骤1：分析问题
    correct_func = debug_task_11()
    
    # 步骤2：测试改进方案
    if os.environ.get('OPENAI_API_KEY'):
        success = test_improved_openai_btp()
        if success:
            print("\n🎉 问题已解决！现在可以运行完整的BTP实验了")
        else:
            print("\n💡 建议:")
            print("1. 检查提示词是否明确指定了函数名")
            print("2. 增加更多示例和约束条件")
            print("3. 使用更高精度的模型")
    else:
        print("\n⚠️ 未设置OpenAI API密钥，跳过在线测试")
        print("但是我们已经找到了问题的根本原因：函数名不匹配")
        print("解决方案：在提示词中明确指定正确的函数名")


if __name__ == "__main__":
    main() 