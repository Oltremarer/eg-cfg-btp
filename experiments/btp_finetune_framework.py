#!/usr/bin/env python3
"""
BTP (Beam Search + Testing + Prioritized Experience Replay) 微调框架
实现论文Algorithm 1中的完整BTP算法，支持丰富的命令行参数配置

支持的采样方式:
1. Power Sampling: P(i) = pi^α / Σ pk^α
2. Rank Sampling: pi = 1/rank(i)
"""

import os
import sys
import json
import argparse
import random
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque


class P2ValueCalculator:
    """P2Value = α × possibility + (1-α) × pass_rate"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def calculate(self, possibility: float, pass_rate: float) -> float:
        return self.alpha * possibility + (1 - self.alpha) * pass_rate


class PrioritizedSampler:
    """支持两种采样方式的优先采样器"""
    
    def __init__(self, method: str = "power", alpha: float = 1.0):
        self.method = method
        self.alpha = alpha
    
    def sample(self, experiences: List[Dict], batch_size: int) -> List[Dict]:
        """根据P2Value进行优先采样"""
        if not experiences or batch_size <= 0:
            return []
        
        if self.method == "power":
            return self._power_sampling(experiences, batch_size)
        elif self.method == "rank":
            return self._rank_sampling(experiences, batch_size)
        else:
            raise ValueError(f"Unsupported sampling method: {self.method}")
    
    def _power_sampling(self, experiences: List[Dict], batch_size: int) -> List[Dict]:
        """幂次采样: P(i) = pi^α / Σ pk^α"""
        import numpy as np
        
        p2values = [exp['p2value'] for exp in experiences]
        p2values = np.maximum(p2values, 1e-8)  # 避免除零
        
        # 计算采样概率
        powered = np.power(p2values, self.alpha)
        probabilities = powered / np.sum(powered)
        
        # 采样
        indices = np.random.choice(
            len(experiences), 
            size=min(batch_size, len(experiences)),
            p=probabilities, 
            replace=False
        )
        
        return [experiences[i] for i in indices]
    
    def _rank_sampling(self, experiences: List[Dict], batch_size: int) -> List[Dict]:
        """排名采样: pi = 1/rank(i)"""
        import numpy as np
        
        # 按P2Value排序
        sorted_exp = sorted(experiences, key=lambda x: x['p2value'], reverse=True)
        
        # 计算排名概率
        ranks = np.arange(1, len(sorted_exp) + 1)
        probabilities = 1.0 / ranks
        probabilities = probabilities / np.sum(probabilities)
        
        # 采样
        indices = np.random.choice(
            len(sorted_exp),
            size=min(batch_size, len(sorted_exp)),
            p=probabilities,
            replace=False
        )
        
        return [sorted_exp[i] for i in indices]


class ExperienceBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.p2calc = P2ValueCalculator()
    
    def add(self, experience: Dict):
        """添加经验"""
        if 'p2value' not in experience:
            experience['p2value'] = self.p2calc.calculate(
                experience['possibility'], 
                experience['pass_rate']
            )
        self.buffer.append(experience)
    
    def get_all(self) -> List[Dict]:
        return list(self.buffer)
    
    def get_stats(self) -> Dict:
        if not self.buffer:
            return {'total': 0}
        
        experiences = list(self.buffer)
        p2values = [exp['p2value'] for exp in experiences]
        pass_rates = [exp['pass_rate'] for exp in experiences]
        
        return {
            'total': len(experiences),
            'avg_p2value': sum(p2values) / len(p2values),
            'avg_pass_rate': sum(pass_rates) / len(pass_rates),
            'perfect_solutions': sum(1 for exp in experiences if exp['pass_rate'] >= 1.0)
        }


class BTPFramework:
    """BTP微调框架主类"""
    
    def __init__(self, 
                 source_model: str,
                 target_model: Optional[str] = None,
                 dataset: str = "mbpp",
                 sampling_method: str = "power",
                 sampling_alpha: float = 1.0,
                 p2value_alpha: float = 0.5,
                 use_lora: bool = True,
                 lora_config: Optional[Dict] = None):
        
        self.source_model = source_model
        self.target_model = target_model or source_model
        self.dataset = dataset
        
        # 初始化组件
        self.p2calc = P2ValueCalculator(p2value_alpha)
        self.sampler = PrioritizedSampler(sampling_method, sampling_alpha)
        self.buffer = ExperienceBuffer()
        
        # LoRA配置
        self.use_lora = use_lora
        self.lora_config = lora_config or {
            'r': 16,
            'alpha': 32,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        }
        
        print(f"BTP框架初始化完成:")
        print(f"  源模型: {self.source_model}")
        print(f"  目标模型: {self.target_model}")
        print(f"  采样方法: {sampling_method}")
        print(f"  采样α: {sampling_alpha}")
        print(f"  P2Value α: {p2value_alpha}")
        print(f"  使用LoRA: {use_lora}")
    
    def phase1_beam_search_sampling(self, 
                                   problems: List[Dict], 
                                   num_beams: int = 5) -> None:
        """阶段1: Beam Search采样"""
        print(f"阶段1: Beam Search采样 (beams={num_beams})")
        
        for i, problem in enumerate(problems):
            print(f"处理问题 {i+1}/{len(problems)}")
            
            # 模拟beam search生成
            candidates = self._simulate_beam_generation(problem, num_beams)
            
            # 测试每个候选
            for j, candidate in enumerate(candidates):
                # 模拟测试结果
                pass_rate = self._simulate_testing(candidate['code'], problem)
                
                experience = {
                    'problem_id': problem.get('task_id', i),
                    'problem_text': problem.get('text', ''),
                    'code': candidate['code'],
                    'possibility': candidate['possibility'],
                    'pass_rate': pass_rate,
                    'beam_rank': j,
                    'source_model': self.source_model
                }
                
                self.buffer.add(experience)
        
        print(f"阶段1完成，收集了 {len(self.buffer.get_all())} 个经验")
    
    def phase2_pper_training(self, 
                           n_iterations: int = 3,
                           batch_size: int = 100,
                           training_config: Optional[Dict] = None) -> None:
        """阶段2: PPER训练"""
        print(f"阶段2: PPER训练 ({n_iterations}轮迭代)")
        
        training_config = training_config or {
            'learning_rate': 1e-4,
            'num_epochs': 1,
            'batch_size': 2,
            'gradient_accumulation_steps': 8
        }
        
        for iteration in range(n_iterations):
            print(f"\n迭代 {iteration + 1}/{n_iterations}")
            
            # 优先采样
            all_experiences = self.buffer.get_all()
            sampled = self.sampler.sample(all_experiences, batch_size)
            
            print(f"从 {len(all_experiences)} 个经验中采样了 {len(sampled)} 个")
            
            # 模拟微调过程
            self._simulate_finetuning(sampled, training_config, iteration)
            
            # 显示采样统计
            self._show_sampling_stats(sampled, all_experiences)
    
    def _simulate_beam_generation(self, problem: Dict, num_beams: int) -> List[Dict]:
        """模拟beam search生成"""
        candidates = []
        
        for i in range(num_beams):
            # 模拟生成的代码和概率
            code = f"# Generated solution {i+1} for problem\ndef solution():\n    return 'mock'"
            possibility = 1.0 / (i + 1)  # 简单的排名概率
            
            candidates.append({
                'code': code,
                'possibility': possibility
            })
        
        return candidates
    
    def _simulate_testing(self, code: str, problem: Dict) -> float:
        """模拟代码测试"""
        # 简单模拟：随机生成通过率
        return random.uniform(0.0, 1.0)
    
    def _simulate_finetuning(self, 
                           experiences: List[Dict], 
                           config: Dict, 
                           iteration: int) -> None:
        """模拟微调过程"""
        print(f"  模拟微调配置:")
        print(f"    学习率: {config['learning_rate']}")
        print(f"    训练轮数: {config['num_epochs']}")
        print(f"    批大小: {config['batch_size']}")
        print(f"    LoRA: {self.use_lora}")
        
        if self.use_lora:
            print(f"    LoRA r: {self.lora_config['r']}")
            print(f"    LoRA α: {self.lora_config['alpha']}")
        
        # 模拟训练时间
        import time
        time.sleep(0.5)  # 模拟训练延迟
        
        print(f"  ✓ 迭代 {iteration + 1} 微调完成")
    
    def _show_sampling_stats(self, sampled: List[Dict], all_exp: List[Dict]) -> None:
        """显示采样统计"""
        if not sampled:
            return
        
        sampled_p2values = [exp['p2value'] for exp in sampled]
        all_p2values = [exp['p2value'] for exp in all_exp]
        
        print(f"  采样统计:")
        print(f"    采样平均P2Value: {sum(sampled_p2values)/len(sampled_p2values):.4f}")
        print(f"    全体平均P2Value: {sum(all_p2values)/len(all_p2values):.4f}")
        print(f"    采样中完美解决方案: {sum(1 for exp in sampled if exp['pass_rate'] >= 1.0)}")
    
    def run_experiment(self, 
                      max_problems: int = 50,
                      num_beams: int = 5,
                      n_iterations: int = 2,
                      batch_size: int = 50,
                      output_dir: str = "./btp_results") -> Dict:
        """运行完整BTP实验"""
        
        print("=" * 60)
        print("BTP微调实验")
        print("=" * 60)
        
        # 模拟加载数据集
        problems = self._load_mock_dataset(max_problems)
        print(f"加载了 {len(problems)} 个问题")
        
        # 阶段1: Beam Search + Testing
        self.phase1_beam_search_sampling(problems, num_beams)
        
        initial_stats = self.buffer.get_stats()
        print(f"\n初始统计:")
        for k, v in initial_stats.items():
            print(f"  {k}: {v}")
        
        # 阶段2: PPER微调
        self.phase2_pper_training(n_iterations, batch_size)
        
        # 保存结果
        results = {
            'experiment_type': 'BTP_FineTune',
            'source_model': self.source_model,
            'target_model': self.target_model,
            'sampling_method': self.sampler.method,
            'sampling_alpha': self.sampler.alpha,
            'p2value_alpha': self.p2calc.alpha,
            'max_problems': max_problems,
            'num_beams': num_beams,
            'n_iterations': n_iterations,
            'batch_size': batch_size,
            'use_lora': self.use_lora,
            'lora_config': self.lora_config,
            'initial_stats': initial_stats,
            'final_stats': self.buffer.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到文件
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, 
                                 f"btp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n实验结果已保存到: {result_file}")
        return results
    
    def _load_mock_dataset(self, max_problems: int) -> List[Dict]:
        """加载模拟数据集"""
        problems = []
        for i in range(max_problems):
            problems.append({
                'task_id': i,
                'text': f'模拟编程问题 {i+1}',
                'test_list': [f'assert solution() == expected_{i}']
            })
        return problems


def main():
    parser = argparse.ArgumentParser(
        description='BTP微调实验框架 - 支持论文中的两种采样方式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
采样方式详解:

1. 幂次采样 (Power Sampling):
   P(i) = pi^α / Σ pk^α
   - α > 1: 更倾向于高P2Value的经验
   - α = 1: 按P2Value比例采样  
   - α < 1: 更平滑的采样分布

2. 排名采样 (Rank Sampling):
   pi = 1/rank(i)
   - 基于P2Value排名的倒数
   - 对异常值更稳健

使用示例:

1. 幂次采样实验:
   python btp_finetune_framework.py \\
     --source-model deepseek-ai/deepseek-coder-1.3b-instruct \\
     --sampling-method power \\
     --sampling-alpha 1.2

2. 排名采样实验:
   python btp_finetune_framework.py \\
     --source-model deepseek-ai/deepseek-coder-6.7b-instruct \\
     --sampling-method rank

3. 调整P2Value权重 (更重视通过率):
   python btp_finetune_framework.py \\
     --source-model deepseek-ai/deepseek-coder-1.3b-instruct \\
     --p2value-alpha 0.2 \\
     --sampling-alpha 1.5

4. 不同源模型和目标模型:
   python btp_finetune_framework.py \\
     --source-model deepseek-ai/deepseek-coder-6.7b-instruct \\
     --target-model codellama/CodeLlama-7b-Instruct-hf \\
     --sampling-method power \\
     --sampling-alpha 0.8

5. 调整LoRA参数:
   python btp_finetune_framework.py \\
     --source-model deepseek-ai/deepseek-coder-1.3b-instruct \\
     --lora-r 32 \\
     --lora-alpha 64 \\
     --learning-rate 2e-4
""")
    
    # 模型参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--source-model', required=True,
                           help='用于初始beam search生成的源模型路径')
    model_group.add_argument('--target-model', 
                           help='用于微调的目标模型路径（默认与源模型相同）')
    
    # 数据集参数
    data_group = parser.add_argument_group('数据集参数')
    data_group.add_argument('--dataset', default='mbpp',
                          help='数据集名称 (默认: mbpp)')
    data_group.add_argument('--max-problems', type=int, default=50,
                          help='处理的最大问题数量')
    
    # BTP算法参数
    algo_group = parser.add_argument_group('BTP算法参数')
    algo_group.add_argument('--num-beams', type=int, default=5,
                          help='Beam Search的beam数量')
    algo_group.add_argument('--n-iterations', type=int, default=3,
                          help='PPER训练迭代次数')
    algo_group.add_argument('--batch-size', type=int, default=100,
                          help='每次迭代采样的经验数量')
    
    # 采样参数 (核心配置)
    sampling_group = parser.add_argument_group('采样参数 (核心配置)')
    sampling_group.add_argument('--sampling-method', default='power',
                              choices=['power', 'rank'],
                              help='采样方法: power(幂次) 或 rank(排名)')
    sampling_group.add_argument('--sampling-alpha', type=float, default=1.0,
                              help='采样参数α，控制采样倾向性')
    sampling_group.add_argument('--p2value-alpha', type=float, default=0.5,
                              help='P2Value权重α (0=仅通过率, 1=仅可能性)')
    
    # LoRA微调参数
    lora_group = parser.add_argument_group('LoRA微调参数')
    lora_group.add_argument('--use-lora', action='store_true', default=True,
                          help='使用LoRA进行高效微调')
    lora_group.add_argument('--lora-r', type=int, default=16,
                          help='LoRA rank (秩)')
    lora_group.add_argument('--lora-alpha', type=int, default=32,
                          help='LoRA缩放参数')
    lora_group.add_argument('--lora-dropout', type=float, default=0.1,
                          help='LoRA dropout率')
    lora_group.add_argument('--lora-target-modules', nargs='+', 
                          default=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                          help='LoRA目标模块')
    
    # 训练参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--learning-rate', type=float, default=1e-4,
                           help='微调学习率')
    train_group.add_argument('--num-epochs', type=int, default=1,
                           help='每次迭代的训练轮数')
    train_group.add_argument('--per-device-batch-size', type=int, default=2,
                           help='每设备批大小')
    train_group.add_argument('--gradient-accumulation-steps', type=int, default=8,
                           help='梯度累积步数')
    train_group.add_argument('--warmup-steps', type=int, default=100,
                           help='预热步数')
    train_group.add_argument('--weight-decay', type=float, default=0.01,
                           help='权重衰减')
    
    # 输出和实验参数
    output_group = parser.add_argument_group('输出和实验参数')
    output_group.add_argument('--output-dir', default='./btp_results',
                            help='结果输出目录')
    output_group.add_argument('--checkpoint-dir', default='./btp_checkpoints',
                            help='模型检查点保存目录')
    output_group.add_argument('--experiment-name', 
                            help='实验名称（用于结果文件命名）')
    output_group.add_argument('--seed', type=int, default=42,
                            help='随机种子')
    output_group.add_argument('--debug', action='store_true',
                            help='启用调试模式')
    
    args = parser.parse_args()
    
    print("🚀 BTP微调实验框架")
    print("=" * 60)
    print("📊 实验配置:")
    print(f"  🤖 源模型: {args.source_model}")
    print(f"  🎯 目标模型: {args.target_model or '与源模型相同'}")
    print(f"  📚 数据集: {args.dataset} ({args.max_problems} 问题)")
    print()
    print("🎲 采样配置:")
    print(f"  📈 采样方法: {args.sampling_method}")
    if args.sampling_method == 'power':
        print(f"       公式: P(i) = pi^{args.sampling_alpha} / Σ pk^{args.sampling_alpha}")
    else:
        print(f"       公式: pi = 1/rank(i)")
    print(f"  ⚖️  P2Value α: {args.p2value_alpha} (可能性 vs 通过率权重)")
    print()
    print("🔧 训练配置:")
    print(f"  🌟 Beam数量: {args.num_beams}")
    print(f"  🔄 迭代次数: {args.n_iterations}")
    print(f"  📦 批大小: {args.batch_size}")
    print(f"  📚 学习率: {args.learning_rate}")
    print(f"  🎪 使用LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"       r={args.lora_r}, α={args.lora_alpha}, dropout={args.lora_dropout}")
    print()
    
    # 展示参数组合的影响
    print("💡 参数说明:")
    
    if args.sampling_method == 'power':
        if args.sampling_alpha > 1.0:
            print(f"  📈 采样α={args.sampling_alpha} > 1.0: 强烈偏向高P2Value经验")
        elif args.sampling_alpha == 1.0:
            print(f"  📊 采样α={args.sampling_alpha} = 1.0: 按P2Value比例采样")
        else:
            print(f"  📉 采样α={args.sampling_alpha} < 1.0: 更平滑的采样分布")
    else:
        print(f"  🏆 排名采样: 基于P2Value排名，对异常值更稳健")
    
    if args.p2value_alpha > 0.5:
        print(f"  🎯 P2Value α={args.p2value_alpha} > 0.5: 更重视生成可能性")
    elif args.p2value_alpha == 0.5:
        print(f"  ⚖️  P2Value α={args.p2value_alpha} = 0.5: 平衡可能性和通过率")
    else:
        print(f"  ✅ P2Value α={args.p2value_alpha} < 0.5: 更重视测试通过率")
    
    print()
    print("🎯 推荐配置组合:")
    print("  💪 激进策略: --sampling-method power --sampling-alpha 1.5 --p2value-alpha 0.3")
    print("  🎯 平衡策略: --sampling-method power --sampling-alpha 1.0 --p2value-alpha 0.5") 
    print("  🛡️  保守策略: --sampling-method rank --p2value-alpha 0.7")
    
    print()
    print("⚠️  注意: 这是实验框架演示版本")
    print("   真实实现需要集成transformers、peft等库进行实际微调")
    print("   当前版本主要展示BTP算法的参数配置选项")
    
    # 生成实验配置文件
    config = {
        'experiment_name': args.experiment_name or f"btp_{args.sampling_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'source_model': args.source_model,
        'target_model': args.target_model,
        'sampling_config': {
            'method': args.sampling_method,
            'alpha': args.sampling_alpha
        },
        'p2value_config': {
            'alpha': args.p2value_alpha
        },
        'btp_config': {
            'num_beams': args.num_beams,
            'n_iterations': args.n_iterations,
            'batch_size': args.batch_size
        },
        'lora_config': {
            'use_lora': args.use_lora,
            'r': args.lora_r,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'target_modules': args.lora_target_modules
        } if args.use_lora else None,
        'training_config': {
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'per_device_batch_size': args.per_device_batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'warmup_steps': args.warmup_steps,
            'weight_decay': args.weight_decay
        },
        'dataset': args.dataset,
        'max_problems': args.max_problems,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存配置
    os.makedirs(args.output_dir, exist_ok=True)
    config_file = os.path.join(args.output_dir, f"{config['experiment_name']}_config.json")
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 实验配置已保存到: {config_file}")
    print("   可用此配置文件驱动真实的BTP微调实验")


if __name__ == "__main__":
    main() 