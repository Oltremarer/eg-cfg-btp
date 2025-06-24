#!/usr/bin/env python3
"""
运行OpenAI版本的BTP实验
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 设置环境变量（如果需要）
# 你需要设置你的OpenAI API密钥
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'  # 替换为你的实际API密钥

def run_openai_btp_experiment():
    """运行OpenAI BTP实验的简化版本"""
    print("=" * 60)
    print("开始运行OpenAI BTP实验")
    print("=" * 60)
    
    try:
        from step2_btp_experiment_openai import BTPOpenAIExperiment
        
        # 实验配置
        config = {
            'model_name': 'gpt-3.5-turbo',  # 或者 'gpt-4' 如果你有访问权限
            'dataset': 'mbpp',
            'api_key': os.environ.get('OPENAI_API_KEY'),
            'base_url': None,  # 使用默认OpenAI API
            'max_problems': 10,  # 先试验少量问题
            'num_candidates': 3  # 每个问题生成3个候选解决方案
        }
        
        print(f"使用配置:")
        print(f"  模型: {config['model_name']}")
        print(f"  数据集: {config['dataset']}")
        print(f"  问题数量: {config['max_problems']}")
        print(f"  候选数量: {config['num_candidates']}")
        
        # 检查API密钥
        if not config['api_key'] or config['api_key'] == 'your-api-key-here':
            print("\n⚠️  警告: 请设置正确的OpenAI API密钥！")
            print("方法1: 设置环境变量 OPENAI_API_KEY")
            print("方法2: 修改此脚本中的 os.environ['OPENAI_API_KEY'] 行")
            return
        
        # 创建实验实例
        experiment = BTPOpenAIExperiment(
            model_name=config['model_name'],
            dataset=config['dataset'],
            api_key=config['api_key'],
            base_url=config['base_url']
        )
        
        # 运行实验
        output_dir = f"experiments/results/btp_openai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = experiment.run_experiment(
            output_dir=output_dir,
            collect_problems=config['max_problems'],
            num_candidates=config['num_candidates']
        )
        
        print("\n" + "=" * 60)
        print("实验完成！")
        print("=" * 60)
        print(f"总体成功率: {results['success_rate']:.2%}")
        print(f"处理的问题数: {results['problems_processed']}")
        print(f"每个问题的候选数: {results['candidates_per_problem']}")
        
        if 'buffer_stats' in results:
            stats = results['buffer_stats']
            print(f"总经验数: {stats.get('total_experiences', 0)}")
            print(f"完全成功率: {stats.get('fully_passed_rate', 0):.2%}")
            print(f"增强解决方案比例: {stats.get('enhanced_rate', 0):.2%}")
        
        print(f"\n详细结果已保存到目录: {output_dir}")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖已正确安装")
    except Exception as e:
        print(f"实验运行出错: {e}")
        import traceback
        traceback.print_exc()


def run_baseline_comparison():
    """运行基线对比实验"""
    print("\n" + "=" * 60)
    print("运行基线对比实验（不使用BTP）")
    print("=" * 60)
    
    try:
        from eg_cfg.openai_utils import OpenAIClient
        from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests
        
        # 加载数据和模型
        problems = load_mbpp_problems()
        client = OpenAIClient(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model='gpt-3.5-turbo'
        )
        
        # 基线实验：直接生成单个解决方案
        success_count = 0
        total_count = 0
        max_problems = 10
        
        problems_list = list(problems.items())[:max_problems]
        
        for task_id, problem in problems_list:
            try:
                prompt = f"""请解决以下编程问题：

问题描述：
{problem['text']}

请提供一个完整的Python函数实现：

```python
"""
                
                # 生成单个解决方案
                solutions = client.generate_code(prompt, n=1, temperature=0.7)
                
                if solutions:
                    code = solutions[0]
                    test_results = run_tests(code, problem['test_list'])
                    passed_tests = sum(1 for r in test_results.values() if r.get('result', False))
                    total_tests = len(test_results)
                    
                    if passed_tests == total_tests and total_tests > 0:
                        success_count += 1
                        print(f"✓ 问题 {task_id}: 成功")
                    else:
                        print(f"✗ 问题 {task_id}: 失败 ({passed_tests}/{total_tests})")
                else:
                    print(f"✗ 问题 {task_id}: 生成失败")
                
                total_count += 1
                
            except Exception as e:
                print(f"✗ 问题 {task_id}: 错误 - {e}")
                total_count += 1
        
        baseline_success_rate = success_count / total_count if total_count > 0 else 0
        print(f"\n基线成功率: {baseline_success_rate:.2%} ({success_count}/{total_count})")
        
        return baseline_success_rate
        
    except Exception as e:
        print(f"基线实验出错: {e}")
        return 0.0


def main():
    """主函数"""
    print("OpenAI BTP实验运行器")
    print("此脚本将运行使用OpenAI GPT的BTP实验")
    
    # 检查环境
    if not os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY') == 'your-api-key-here':
        print("\n请先设置OpenAI API密钥:")
        print("1. 获取API密钥: https://platform.openai.com/api-keys")
        print("2. 设置环境变量: export OPENAI_API_KEY='your-api-key'")
        print("3. 或者修改此脚本中的API密钥设置")
        return
    
    # 运行基线实验
    print("\n第一步: 运行基线实验...")
    baseline_rate = run_baseline_comparison()
    
    # 运行BTP实验
    print("\n第二步: 运行BTP实验...")
    run_openai_btp_experiment()
    
    print(f"\n实验对比:")
    print(f"基线成功率: {baseline_rate:.2%}")
    print("BTP成功率: 请查看上面的实验结果")


if __name__ == "__main__":
    main() 