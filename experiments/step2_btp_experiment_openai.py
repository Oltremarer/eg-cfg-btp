#!/usr/bin/env python3
"""
实验二：OpenAI GPT版本的BTP管道有效性验证
BTP = Beam Search + Testing + Prioritized Experience Replay
使用OpenAI GPT API进行代码生成
"""

import os
import sys
import json
import numpy as np
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
from datetime import datetime
import math
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from eg_cfg.mbpp_utils import load_mbpp_problems, run_tests
from eg_cfg.openai_utils import OpenAIClient, OpenAIInferenceError


class P2ValueCalculator:
    """计算P2Value指标：综合生成概率和测试通过率"""
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    
    def calculate_p2value(self, possibility_score, passed_tests, total_tests):
        """计算P2Value指标（OpenAI版本，不直接使用log_prob）"""
        # 标准化可能性分数（基于温度和随机性）
        possibility = min(possibility_score, 1.0)
        
        # 计算通过率
        pass_rate = passed_tests / max(total_tests, 1)
        
        # 综合分数
        p2value = self.alpha * possibility + (1 - self.alpha) * pass_rate
        return p2value, possibility, pass_rate


class ExperienceReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.p2calculator = P2ValueCalculator()
    
    def add_experience(self, experience):
        """添加经验到缓冲区"""
        p2value, possibility, pass_rate = self.p2calculator.calculate_p2value(
            experience['possibility_score'],
            experience['passed_tests'],
            experience['total_tests']
        )
        
        experience['p2value'] = p2value
        experience['possibility_score'] = possibility
        experience['pass_rate_score'] = pass_rate
        
        self.buffer.append(experience)
    
    def sample_prioritized(self, batch_size, failed_only=True):
        """基于P2Value优先采样"""
        if failed_only:
            available = [exp for exp in self.buffer if exp['pass_rate_score'] < 1.0]
        else:
            available = list(self.buffer)
        
        if not available:
            return []
        
        # 按P2Value排序
        available.sort(key=lambda x: x['p2value'], reverse=True)
        return available[:min(batch_size, len(available))]


class BTPOpenAIExperiment:
    """BTP实验主类 - OpenAI版本"""
    
    def __init__(self, model_name: str, dataset: str, api_key: str = None, base_url: str = None):
        self.model_name = model_name
        self.dataset = dataset
        
        print(f"初始化OpenAI客户端，模型: {model_name}")
        self.client = OpenAIClient(api_key=api_key, base_url=base_url, model=model_name)
        
        print(f"加载数据集: {dataset}")
        if dataset == "mbpp":
            self.problems = load_mbpp_problems()
        else:
            raise ValueError(f"不支持的数据集: {dataset}")
        
        self.replay_buffer = ExperienceReplayBuffer()
        print(f"已加载 {len(self.problems)} 个问题")
    
    def generate_multiple_solutions(self, prompt, num_candidates=5, temperature=0.8, max_tokens=512):
        """使用OpenAI API生成多个代码解决方案"""
        try:
            solutions = self.client.generate_code(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_candidates
            )
            
            results = []
            for i, code in enumerate(solutions):
                # 为OpenAI生成的代码计算一个可能性分数
                # 基于温度和候选排名的启发式评分
                possibility_score = max(0.1, 1.0 - (temperature * 0.5) - (i * 0.1))
                
                results.append({
                    'code': code,
                    'possibility_score': possibility_score,
                    'generation_rank': i + 1,
                    'temperature': temperature
                })
            
            return results
            
        except OpenAIInferenceError as e:
            print(f"OpenAI API错误: {e}")
            return []
        except Exception as e:
            print(f"生成解决方案时出错: {e}")
            return []
    
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
                    'fully_passed': passed_tests == total_tests and total_tests > 0
                })
                
            except Exception as e:
                results.append({
                    **sol,
                    'test_results': {},
                    'passed_tests': 0,
                    'total_tests': len(problem.get('test_list', [])),
                    'pass_rate': 0.0,
                    'fully_passed': False,
                    'error': str(e)
                })
        
        return results
    
    def extract_function_name(self, test_list):
        """从测试用例中提取函数名"""
        import re
        for test in test_list:
            # 匹配 assert function_name(...) == ... 格式
            match = re.search(r'assert\s+(\w+)\s*\(', test)
            if match:
                return match.group(1)
        return None
    
    def format_prompt(self, problem):
        """格式化问题为提示词"""
        # 从测试用例中提取函数名（通用方法）
        function_name = self.extract_function_name(problem.get('test_list', []))
        
        # 构建测试用例说明
        test_examples = "\n".join([f"  {test}" for test in problem.get('test_list', [])])
        
        return f"""请解决以下编程问题：

问题描述：
{problem['text']}

测试用例（你的函数必须通过这些测试）：
{test_examples}

要求：
1. 请提供一个完整的Python函数实现
2. 函数必须能够通过所有给定的测试用例
3. 代码应该简洁、高效且正确
4. 请确保函数名与测试用例中的函数名一致

请在代码块中提供解决方案：

```python
"""
    
    def format_prioritized_prompt(self, problem, previous_experiences):
        """基于经验回放格式化增强提示词"""
        base_prompt = self.format_prompt(problem)
        
        if not previous_experiences:
            return base_prompt
        
        # 添加经验学习部分
        experience_section = "\n\n参考以下类似问题的解决经验：\n"
        
        for i, exp in enumerate(previous_experiences[:3]):  # 只使用前3个经验
            if exp['pass_rate_score'] > 0.5:  # 只使用相对成功的经验
                experience_section += f"\n经验 {i+1}（通过率: {exp['pass_rate_score']:.2f}）：\n"
                experience_section += f"```python\n{exp['code'][:200]}...\n```\n"
        
        enhanced_prompt = base_prompt + experience_section + "\n基于以上经验，请提供更好的解决方案：\n\n```python\n"
        return enhanced_prompt
    
    def run_experiment(self, output_dir, collect_problems=50, num_candidates=5):
        """运行完整的BTP实验"""
        print("=" * 60)
        print("BTP OpenAI实验开始")
        print("=" * 60)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 收集经验数据
        success_rate = self.collect_experiences(
            max_problems=collect_problems,
            num_candidates=num_candidates,
            use_prioritized=True
        )
        
        # 分析缓冲区
        buffer_stats = self.analyze_buffer()
        
        # 保存实验结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'experiment_type': 'BTP_OpenAI',
            'model_name': self.model_name,
            'dataset': self.dataset,
            'timestamp': timestamp,
            'success_rate': success_rate,
            'problems_processed': collect_problems,
            'candidates_per_problem': num_candidates,
            'buffer_stats': buffer_stats,
            'hyperparameters': {
                'alpha': 0.5,
                'temperature_base': 0.8,
                'temperature_enhanced': 0.7,
                'max_tokens': 512
            }
        }
        
        # 保存详细结果
        result_file = output_path / f"btp_openai_results_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n实验结果已保存到: {result_file}")
        return results
    
    def collect_experiences(self, max_problems=50, num_candidates=5, use_prioritized=True):
        """收集经验数据"""
        print(f"开始收集经验数据... (问题数量: {max_problems}, 候选数量: {num_candidates})")
        
        problems_list = list(self.problems.items())[:max_problems]
        
        total_success = 0
        total_attempts = 0
        
        for task_id, problem in tqdm(problems_list, desc="处理问题"):
            try:
                print(f"\n处理问题 {task_id}: {problem['text'][:50]}...")
                
                # 第一轮：基础生成
                base_prompt = self.format_prompt(problem)
                solutions = self.generate_multiple_solutions(
                    base_prompt, 
                    num_candidates=num_candidates,
                    temperature=0.8
                )
                
                if not solutions:
                    continue
                
                # 测试解决方案
                tested_solutions = self.test_code_solutions(problem, solutions)
                
                # 添加到经验缓冲区
                for sol in tested_solutions:
                    experience = {
                        'task_id': task_id,
                        'problem_text': problem['text'],
                        'code': sol['code'],
                        'possibility_score': sol['possibility_score'],
                        'passed_tests': sol['passed_tests'],
                        'total_tests': sol['total_tests'],
                        'fully_passed': sol['fully_passed']
                    }
                    self.replay_buffer.add_experience(experience)
                
                # 统计成功率
                if any(sol['fully_passed'] for sol in tested_solutions):
                    total_success += 1
                total_attempts += 1
                
                # 第二轮：如果没有完全成功且启用优先经验回放
                if use_prioritized and not any(sol['fully_passed'] for sol in tested_solutions):
                    print("  基础生成未完全成功，尝试经验增强生成...")
                    
                    # 获取相关经验
                    similar_experiences = self.replay_buffer.sample_prioritized(
                        batch_size=5, 
                        failed_only=False
                    )
                    
                    if similar_experiences:
                        enhanced_prompt = self.format_prioritized_prompt(problem, similar_experiences)
                        enhanced_solutions = self.generate_multiple_solutions(
                            enhanced_prompt,
                            num_candidates=3,
                            temperature=0.7  # 稍微降低温度
                        )
                        
                        if enhanced_solutions:
                            enhanced_tested = self.test_code_solutions(problem, enhanced_solutions)
                            
                            # 添加增强解决方案到缓冲区
                            for sol in enhanced_tested:
                                experience = {
                                    'task_id': task_id,
                                    'problem_text': problem['text'],
                                    'code': sol['code'],
                                    'possibility_score': sol['possibility_score'] * 1.1,  # 增强解决方案获得小幅加分
                                    'passed_tests': sol['passed_tests'],
                                    'total_tests': sol['total_tests'],
                                    'fully_passed': sol['fully_passed'],
                                    'enhanced': True
                                }
                                self.replay_buffer.add_experience(experience)
                            
                            # 检查增强解决方案是否成功
                            if any(sol['fully_passed'] for sol in enhanced_tested):
                                total_success += 1
                                print("  经验增强生成成功！")
                
                # 添加延迟以避免API限制
                time.sleep(1)
                
            except Exception as e:
                print(f"处理问题 {task_id} 时出错: {e}")
                continue
        
        success_rate = total_success / total_attempts if total_attempts > 0 else 0
        print(f"\n数据收集完成！成功率: {success_rate:.2%} ({total_success}/{total_attempts})")
        return success_rate
    
    def analyze_buffer(self):
        """分析经验缓冲区"""
        if not self.replay_buffer.buffer:
            return {}
        
        experiences = list(self.replay_buffer.buffer)
        
        # 基础统计
        total_experiences = len(experiences)
        fully_passed_count = sum(1 for exp in experiences if exp.get('fully_passed', False))
        enhanced_count = sum(1 for exp in experiences if exp.get('enhanced', False))
        
        # P2Value分布
        p2values = [exp['p2value'] for exp in experiences]
        pass_rates = [exp['pass_rate_score'] for exp in experiences]
        possibility_scores = [exp['possibility_score'] for exp in experiences]
        
        stats = {
            'total_experiences': total_experiences,
            'fully_passed_count': fully_passed_count,
            'enhanced_count': enhanced_count,
            'fully_passed_rate': fully_passed_count / total_experiences if total_experiences > 0 else 0,
            'enhanced_rate': enhanced_count / total_experiences if total_experiences > 0 else 0,
            'p2value_stats': {
                'mean': np.mean(p2values),
                'std': np.std(p2values),
                'min': np.min(p2values),
                'max': np.max(p2values)
            },
            'pass_rate_stats': {
                'mean': np.mean(pass_rates),
                'std': np.std(pass_rates),
                'min': np.min(pass_rates),
                'max': np.max(pass_rates)
            },
            'possibility_stats': {
                'mean': np.mean(possibility_scores),
                'std': np.std(possibility_scores),
                'min': np.min(possibility_scores),
                'max': np.max(possibility_scores)
            }
        }
        
        print(f"\n缓冲区分析:")
        print(f"  总经验数: {stats['total_experiences']}")
        print(f"  完全成功率: {stats['fully_passed_rate']:.2%}")
        print(f"  增强解决方案比例: {stats['enhanced_rate']:.2%}")
        print(f"  平均P2Value: {stats['p2value_stats']['mean']:.3f}")
        print(f"  平均通过率: {stats['pass_rate_stats']['mean']:.3f}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='BTP OpenAI实验')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', 
                       help='OpenAI模型名称')
    parser.add_argument('--dataset', type=str, default='mbpp', 
                       help='数据集名称')
    parser.add_argument('--problems', type=int, default=50, 
                       help='处理的问题数量')
    parser.add_argument('--candidates', type=int, default=5, 
                       help='每个问题的候选解决方案数量')
    parser.add_argument('--output-dir', type=str, default='experiments/results/btp_openai',
                       help='输出目录')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API密钥')
    parser.add_argument('--base-url', type=str, default=None,
                       help='OpenAI API基础URL')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    print(f"使用模型: {args.model}")
    print(f"处理问题数: {args.problems}")
    print(f"候选数量: {args.candidates}")
    
    # 创建实验实例
    experiment = BTPOpenAIExperiment(
        model_name=args.model,
        dataset=args.dataset,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # 运行实验
    results = experiment.run_experiment(
        output_dir=args.output_dir,
        collect_problems=args.problems,
        num_candidates=args.candidates
    )
    
    print("\n实验完成！")
    print(f"总体成功率: {results['success_rate']:.2%}")


if __name__ == "__main__":
    main() 