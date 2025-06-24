#!/usr/bin/env python3
"""
实验四：超参数敏感性分析
探索关键超参数对模型性能的影响
"""

import os
import sys
import json
import itertools
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from step2_btp_experiment import BTPExperiment, P2ValueCalculator
from eg_cfg.mbpp_utils import load_mbpp_problems


class HyperparameterStudy:
    """超参数敏感性分析"""
    
    def __init__(self, model_name: str, dataset: str = "mbpp"):
        self.model_name = model_name
        self.dataset = dataset
        
        # 定义超参数搜索空间
        self.param_space = {
            'alpha': [0.3, 0.5, 0.7],  # P2Value中的权重参数
            'buffer_size': [1000, 5000, 10000],  # 回放缓冲区大小
            'num_beams': [3, 5, 8],  # Beam search宽度
            'temperature': [0.6, 0.8, 1.0],  # 采样温度
            'batch_size': [10, 20, 50]  # 采样批次大小
        }
        
        print(f"Hyperparameter study for {model_name} on {dataset}")
        print(f"Parameter space: {self.param_space}")
    
    def run_single_config(self, config, problems_subset, max_problems=20):
        """运行单个超参数配置"""
        print(f"Running config: {config}")
        
        try:
            # 创建自定义的BTP实验
            experiment = BTPExperiment(self.model_name, self.dataset)
            
            # 修改超参数
            experiment.replay_buffer.max_size = config['buffer_size']
            experiment.replay_buffer.p2calculator.alpha = config['alpha']
            
            # 收集经验（使用指定的beam数量）
            experiment.collect_experiences(
                max_problems=max_problems, 
                num_beams=config['num_beams']
            )
            
            # 评估性能
            performance = self.evaluate_config_performance(
                experiment, 
                problems_subset, 
                config
            )
            
            return {
                'config': config,
                'performance': performance,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error with config {config}: {e}")
            return {
                'config': config,
                'performance': None,
                'status': 'error',
                'error': str(e)
            }
    
    def evaluate_config_performance(self, experiment, problems_subset, config):
        """评估配置性能"""
        results = []
        
        for task_id, problem in problems_subset[:10]:  # 限制评估问题数量
            try:
                # 从缓冲区采样
                sampled_experiences = experiment.replay_buffer.sample_prioritized(
                    batch_size=config['batch_size'], 
                    failed_only=True
                )
                
                if sampled_experiences:
                    # 计算采样质量指标
                    avg_p2value = sum(exp['p2value'] for exp in sampled_experiences) / len(sampled_experiences)
                    avg_pass_rate = sum(exp['pass_rate_score'] for exp in sampled_experiences) / len(sampled_experiences)
                    
                    results.append({
                        'task_id': task_id,
                        'sampled_count': len(sampled_experiences),
                        'avg_p2value': avg_p2value,
                        'avg_pass_rate': avg_pass_rate
                    })
                
            except Exception as e:
                print(f"Error evaluating {task_id}: {e}")
                continue
        
        if not results:
            return {
                'avg_p2value': 0.0,
                'avg_pass_rate': 0.0,
                'successful_evaluations': 0
            }
        
        # 汇总性能指标
        overall_performance = {
            'avg_p2value': sum(r['avg_p2value'] for r in results) / len(results),
            'avg_pass_rate': sum(r['avg_pass_rate'] for r in results) / len(results),
            'successful_evaluations': len(results),
            'detailed_results': results
        }
        
        return overall_performance
    
    def run_grid_search(self, output_dir, max_configs=20, problems_per_config=15):
        """运行网格搜索"""
        print("Starting hyperparameter grid search...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载问题子集
        problems = load_mbpp_problems()
        problems_list = list(problems.items())[:problems_per_config]
        
        # 生成所有超参数组合
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        all_configs = list(itertools.product(*param_values))
        
        # 限制配置数量
        if len(all_configs) > max_configs:
            import random
            all_configs = random.sample(all_configs, max_configs)
        
        print(f"Testing {len(all_configs)} configurations...")
        
        results = []
        
        for i, config_values in enumerate(tqdm(all_configs, desc="Grid search")):
            config = dict(zip(param_names, config_values))
            
            result = self.run_single_config(config, problems_list, max_problems=10)
            results.append(result)
            
            # 保存中间结果
            if (i + 1) % 5 == 0:
                self.save_intermediate_results(results, output_path / "intermediate_results.json")
        
        # 分析结果
        analysis = self.analyze_grid_search_results(results)
        
        # 保存最终结果
        final_results = {
            'experiment_config': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'param_space': self.param_space,
                'total_configs': len(all_configs),
                'problems_per_config': problems_per_config,
                'timestamp': datetime.now().isoformat()
            },
            'results': results,
            'analysis': analysis
        }
        
        with open(output_path / "hyperparameter_study_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # 打印最佳配置
        self.print_best_configs(analysis)
        
        return final_results
    
    def save_intermediate_results(self, results, filepath):
        """保存中间结果"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def analyze_grid_search_results(self, results):
        """分析网格搜索结果"""
        successful_results = [r for r in results if r['status'] == 'success' and r['performance']]
        
        if not successful_results:
            return {'error': 'No successful configurations'}
        
        # 按性能排序
        successful_results.sort(
            key=lambda x: x['performance']['avg_p2value'], 
            reverse=True
        )
        
        # 分析每个超参数的影响
        param_analysis = self.analyze_parameter_effects(successful_results)
        
        # 找出最佳配置
        best_configs = successful_results[:5]  # Top 5
        
        analysis = {
            'total_configs': len(results),
            'successful_configs': len(successful_results),
            'best_configs': best_configs,
            'parameter_analysis': param_analysis,
            'performance_distribution': {
                'avg_p2value_mean': sum(r['performance']['avg_p2value'] for r in successful_results) / len(successful_results),
                'avg_p2value_std': self.calculate_std([r['performance']['avg_p2value'] for r in successful_results]),
                'avg_pass_rate_mean': sum(r['performance']['avg_pass_rate'] for r in successful_results) / len(successful_results),
                'avg_pass_rate_std': self.calculate_std([r['performance']['avg_pass_rate'] for r in successful_results])
            }
        }
        
        return analysis
    
    def analyze_parameter_effects(self, results):
        """分析各参数的效果"""
        param_effects = {}
        
        for param_name in self.param_space.keys():
            param_performance = {}
            
            for result in results:
                param_value = result['config'][param_name]
                performance = result['performance']['avg_p2value']
                
                if param_value not in param_performance:
                    param_performance[param_value] = []
                param_performance[param_value].append(performance)
            
            # 计算每个参数值的平均性能
            param_avg_performance = {
                value: sum(perfs) / len(perfs) 
                for value, perfs in param_performance.items()
            }
            
            param_effects[param_name] = {
                'value_performance': param_avg_performance,
                'best_value': max(param_avg_performance, key=param_avg_performance.get),
                'performance_range': max(param_avg_performance.values()) - min(param_avg_performance.values())
            }
        
        return param_effects
    
    def calculate_std(self, values):
        """计算标准差"""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def print_best_configs(self, analysis):
        """打印最佳配置"""
        print("\n" + "="*60)
        print("HYPERPARAMETER STUDY RESULTS")
        print("="*60)
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        print(f"\nTested {analysis['total_configs']} configurations")
        print(f"Successful: {analysis['successful_configs']}")
        
        print("\nTOP 3 CONFIGURATIONS:")
        for i, config_result in enumerate(analysis['best_configs'][:3]):
            config = config_result['config']
            performance = config_result['performance']
            
            print(f"\n{i+1}. Configuration:")
            for param, value in config.items():
                print(f"   {param}: {value}")
            print(f"   Performance - P2Value: {performance['avg_p2value']:.4f}, Pass Rate: {performance['avg_pass_rate']:.4f}")
        
        print("\nPARAMETER SENSITIVITY ANALYSIS:")
        for param_name, param_analysis in analysis['parameter_analysis'].items():
            best_value = param_analysis['best_value']
            performance_range = param_analysis['performance_range']
            print(f"\n{param_name}:")
            print(f"   Best value: {best_value}")
            print(f"   Performance range: {performance_range:.4f}")
            print(f"   Value performance: {param_analysis['value_performance']}")
        
        print("\n" + "="*60)
    
    def run_focused_search(self, output_dir, focus_params=None, num_configs=10):
        """运行针对特定参数的搜索"""
        if focus_params is None:
            focus_params = ['alpha', 'num_beams']  # 默认关注这两个参数
        
        print(f"Running focused search on parameters: {focus_params}")
        
        # 创建针对性的参数空间
        focused_space = {param: self.param_space[param] for param in focus_params}
        
        # 其他参数使用默认值
        default_config = {
            'alpha': 0.5,
            'buffer_size': 5000,
            'num_beams': 5,
            'temperature': 0.8,
            'batch_size': 20
        }
        
        # 生成配置
        param_names = list(focused_space.keys())
        param_values = list(focused_space.values())
        focused_configs = list(itertools.product(*param_values))
        
        results = []
        problems = load_mbpp_problems()
        problems_list = list(problems.items())[:15]
        
        for config_values in tqdm(focused_configs, desc="Focused search"):
            config = default_config.copy()
            config.update(dict(zip(param_names, config_values)))
            
            result = self.run_single_config(config, problems_list, max_problems=8)
            results.append(result)
        
        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        focused_results = {
            'focus_params': focus_params,
            'default_config': default_config,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / "focused_search_results.json", 'w') as f:
            json.dump(focused_results, f, indent=2)
        
        return focused_results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Sensitivity Study")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--output_dir", type=str, default="experiments/results/hyperparameter")
    parser.add_argument("--search_type", type=str, choices=["grid", "focused"], default="grid")
    parser.add_argument("--max_configs", type=int, default=15)
    parser.add_argument("--focus_params", type=str, nargs="+", default=["alpha", "num_beams"])
    
    args = parser.parse_args()
    
    study = HyperparameterStudy(args.model_name, args.dataset)
    
    if args.search_type == "grid":
        results = study.run_grid_search(args.output_dir, max_configs=args.max_configs)
    elif args.search_type == "focused":
        results = study.run_focused_search(args.output_dir, focus_params=args.focus_params)
    
    print(f"Hyperparameter study completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 