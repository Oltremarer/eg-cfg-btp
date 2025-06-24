#!/usr/bin/env python3
"""
BTP微调实验运行脚本
支持各种配置选项的命令行调用
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """检查必要的依赖"""
    required_packages = [
        'torch',
        'transformers',
        'peft',
        'datasets',
        'numpy',
        'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请安装缺少的包:")
        print("pip install torch transformers peft datasets numpy tqdm")
        return False
    
    return True


def run_experiment_command(args):
    """构建并执行实验命令"""
    
    # 基础命令
    cmd = [
        sys.executable, 
        "experiments/step2_btp_finetune_experiment.py",
        "--source-model", args.source_model
    ]
    
    # 添加可选参数
    if args.target_model:
        cmd.extend(["--target-model", args.target_model])
    
    cmd.extend([
        "--dataset", args.dataset,
        "--max-problems", str(args.max_problems),
        "--num-beams", str(args.num_beams),
        "--n-iterations", str(args.n_iterations),
        "--batch-size", str(args.batch_size),
        "--sampling-method", args.sampling_method,
        "--sampling-alpha", str(args.sampling_alpha),
        "--p2value-alpha", str(args.p2value_alpha),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--learning-rate", str(args.learning_rate),
        "--num-epochs", str(args.num_epochs),
        "--per-device-batch-size", str(args.per_device_batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--output-dir", args.output_dir,
        "--checkpoint-dir", args.checkpoint_dir,
        "--seed", str(args.seed)
    ])
    
    if args.use_lora:
        cmd.append("--use-lora")
    
    if args.debug:
        cmd.append("--debug")
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"实验执行失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='BTP微调实验运行器')
    
    # 模型相关参数
    parser.add_argument('--source-model', type=str, required=True,
                       help='用于初始生成的源模型路径')
    parser.add_argument('--target-model', type=str, default=None,
                       help='用于微调的目标模型路径（默认与源模型相同）')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='mbpp',
                       help='数据集名称（默认: mbpp）')
    parser.add_argument('--max-problems', type=int, default=50,
                       help='处理的最大问题数量')
    
    # BTP算法参数
    parser.add_argument('--num-beams', type=int, default=5,
                       help='Beam Search的beam数量')
    parser.add_argument('--n-iterations', type=int, default=2,
                       help='PPER训练迭代次数')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='训练批大小')
    
    # 采样参数
    parser.add_argument('--sampling-method', type=str, default='power', 
                       choices=['power', 'rank'],
                       help='采样方法: power (幂次采样) 或 rank (排名采样)')
    parser.add_argument('--sampling-alpha', type=float, default=1.0,
                       help='采样参数α')
    parser.add_argument('--p2value-alpha', type=float, default=0.5,
                       help='P2Value计算参数α')
    
    # LoRA参数
    parser.add_argument('--use-lora', action='store_true', default=True,
                       help='使用LoRA进行高效微调')
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # 训练参数
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='微调学习率')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='每次迭代的训练轮数')
    parser.add_argument('--per-device-batch-size', type=int, default=2,
                       help='每设备批大小')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                       help='梯度累积步数')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='./btp_results',
                       help='结果输出目录')
    parser.add_argument('--checkpoint-dir', type=str, default='./btp_checkpoints',
                       help='模型检查点目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试日志')
    parser.add_argument('--check-deps-only', action='store_true',
                       help='仅检查依赖，不运行实验')
    
    args = parser.parse_args()
    
    print("BTP微调实验运行器")
    print("=" * 50)
    
    # 检查依赖
    print("检查依赖...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ 所有依赖都已安装")
    
    if args.check_deps_only:
        print("仅检查依赖，退出")
        return
    
    # 显示配置
    print(f"\n实验配置:")
    print(f"  源模型: {args.source_model}")
    print(f"  目标模型: {args.target_model or '与源模型相同'}")
    print(f"  数据集: {args.dataset}")
    print(f"  最大问题数: {args.max_problems}")
    print(f"  采样方法: {args.sampling_method}")
    print(f"  采样α: {args.sampling_alpha}")
    print(f"  P2Value α: {args.p2value_alpha}")
    print(f"  使用LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"  LoRA配置: r={args.lora_r}, α={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  迭代次数: {args.n_iterations}")
    print(f"  学习率: {args.learning_rate}")
    
    # 运行实验
    print(f"\n开始运行BTP微调实验...")
    success = run_experiment_command(args)
    
    if success:
        print("\n✓ 实验完成!")
    else:
        print("\n✗ 实验失败!")
        sys.exit(1)


if __name__ == "__main__":
    # 显示使用示例
    if len(sys.argv) == 1:
        print("BTP微调实验运行器")
        print("=" * 50)
        print("\n使用示例:")
        print("\n1. 使用DeepSeek模型 + 幂次采样:")
        print("python experiments/run_btp_finetune_experiment.py \\")
        print("  --source-model deepseek-ai/deepseek-coder-1.3b-instruct \\")
        print("  --sampling-method power \\")
        print("  --sampling-alpha 1.0 \\")
        print("  --max-problems 30")
        
        print("\n2. 使用排名采样:")
        print("python experiments/run_btp_finetune_experiment.py \\")
        print("  --source-model deepseek-ai/deepseek-coder-6.7b-instruct \\")
        print("  --sampling-method rank \\")
        print("  --max-problems 50")
        
        print("\n3. 使用不同的源模型和目标模型:")
        print("python experiments/run_btp_finetune_experiment.py \\")
        print("  --source-model deepseek-ai/deepseek-coder-6.7b-instruct \\")
        print("  --target-model codellama/CodeLlama-7b-Instruct-hf \\")
        print("  --sampling-method power \\")
        print("  --sampling-alpha 0.8")
        
        print("\n4. 调整P2Value权重:")
        print("python experiments/run_btp_finetune_experiment.py \\")
        print("  --source-model deepseek-ai/deepseek-coder-1.3b-instruct \\")
        print("  --p2value-alpha 0.3 \\")
        print("  --sampling-alpha 1.2")
        
        print("\n5. 仅检查依赖:")
        print("python experiments/run_btp_finetune_experiment.py --check-deps-only")
        
        print("\n6. 调试模式:")
        print("python experiments/run_btp_finetune_experiment.py \\")
        print("  --source-model deepseek-ai/deepseek-coder-1.3b-instruct \\")
        print("  --max-problems 10 \\")
        print("  --debug")
        
        print("\n参数说明:")
        print("  采样方法:")
        print("    - power: P(i) = pi^α / Σ pk^α (幂次采样)")
        print("    - rank:  pi = 1/rank(i) (排名采样)")
        print("  采样α: 控制采样倾向性，值越大越偏向高P2Value样本")
        print("  P2Value α: 控制可能性和通过率的权重平衡")
        
        sys.exit(0)
    
    main() 