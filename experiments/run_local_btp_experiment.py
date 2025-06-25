#!/usr/bin/env python3
"""
本地BTP实验快速启动脚本
封装常用配置，方便快速启动实验
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from big_to_small_finetune_experiment import LocalBTPFineTuneExperiment
from transformers import TrainingArguments


def get_model_configs():
    """预定义的模型配置"""
    return {
        "small_to_small": {
            "source_model": "deepseek-ai/deepseek-coder-1.3b-instruct",
            "target_model": "deepseek-ai/deepseek-coder-1.3b-instruct",
            "description": "1.3B模型自举实验"
        },
        "medium_to_small": {
            "source_model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "target_model": "deepseek-ai/deepseek-coder-1.3b-instruct", 
            "description": "6.7B -> 1.3B 推荐配置"
        },
        "medium_to_medium": {
            "source_model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "target_model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "description": "6.7B模型自举实验"
        }
    }


def get_experiment_presets():
    """预定义的实验配置"""
    return {
        "quick": {
            "max_problems": 20,
            "num_beams": 5,
            "n_iterations": 2,
            "batch_size": 16,
            "description": "快速验证（20分钟）"
        },
        "small": {
            "max_problems": 30,
            "num_beams": 6,
            "n_iterations": 3,
            "batch_size": 24,
            "description": "小规模实验（40分钟）"
        },
        "medium": {
            "max_problems": 50,
            "num_beams": 8,
            "n_iterations": 3,
            "batch_size": 32,
            "description": "中等规模实验（70分钟）"
        },
        "large": {
            "max_problems": 100,
            "num_beams": 10,
            "n_iterations": 4,
            "batch_size": 48,
            "description": "大规模实验（2-3小时）"
        }
    }


def create_optimized_training_args(output_dir: str, experiment_size: str):
    """根据实验规模创建优化的训练参数"""
    
    base_config = {
        "output_dir": output_dir,
        "fp16": True,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "report_to": None,
        "save_total_limit": 2,
        "logging_steps": 10,
    }
    
    if experiment_size == "quick":
        config = {
            **base_config,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 20,
            "save_steps": 50,
        }
    elif experiment_size == "small":
        config = {
            **base_config,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 6,
            "learning_rate": 1e-4,
            "warmup_steps": 30,
            "save_steps": 100,
        }
    elif experiment_size == "medium":
        config = {
            **base_config,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "warmup_steps": 50,
            "save_steps": 100,
        }
    else:  # large
        config = {
            **base_config,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 5e-5,
            "warmup_steps": 100,
            "save_steps": 200,
        }
    
    return TrainingArguments(**config)


def main():
    parser = argparse.ArgumentParser(description='本地BTP实验快速启动')
    
    # 预设配置
    parser.add_argument('--model-config', type=str, 
                       choices=list(get_model_configs().keys()),
                       default='medium_to_small',
                       help='模型配置预设')
    
    parser.add_argument('--experiment-size', type=str,
                       choices=list(get_experiment_presets().keys()),
                       default='small',
                       help='实验规模预设')
    
    # 自定义参数（可覆盖预设）
    parser.add_argument('--source-model', type=str, help='源模型路径（覆盖预设）')
    parser.add_argument('--target-model', type=str, help='目标模型路径（覆盖预设）')
    parser.add_argument('--max-problems', type=int, help='问题数量（覆盖预设）')
    parser.add_argument('--num-beams', type=int, help='Beam数量（覆盖预设）')
    parser.add_argument('--batch-size', type=int, help='批次大小（覆盖预设）')
    
    # 采样参数
    parser.add_argument('--sampling-method', type=str, default='power',
                       choices=['power', 'rank'], help='采样方法')
    parser.add_argument('--sampling-alpha', type=float, default=1.5,
                       help='采样α参数')
    parser.add_argument('--p2value-alpha', type=float, default=0.7,
                       help='P2Value α参数（重视通过率）')
    
    # 输出
    parser.add_argument('--output-dir', type=str, 
                       help='输出目录（默认自动生成）')
    
    # 其他
    parser.add_argument('--dry-run', action='store_true',
                       help='只显示配置，不运行实验')
    
    args = parser.parse_args()
    
    # 获取预设配置
    model_configs = get_model_configs()
    experiment_presets = get_experiment_presets()
    
    model_config = model_configs[args.model_config]
    experiment_preset = experiment_presets[args.experiment_size]
    
    # 应用自定义参数（如果提供）
    source_model = args.source_model or model_config['source_model']
    target_model = args.target_model or model_config['target_model']
    max_problems = args.max_problems or experiment_preset['max_problems']
    num_beams = args.num_beams or experiment_preset['num_beams']
    n_iterations = experiment_preset['n_iterations']
    batch_size = args.batch_size or experiment_preset['batch_size']
    
    # 生成输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"./local_btp_results/{args.model_config}_{args.experiment_size}_{timestamp}"
    
    # 显示配置
    print("=" * 80)
    print("🚀 本地BTP实验启动")
    print("=" * 80)
    print(f"📋 模型配置: {args.model_config} - {model_config['description']}")
    print(f"📊 实验规模: {args.experiment_size} - {experiment_preset['description']}")
    print()
    print("🔧 详细配置:")
    print(f"  📊 源模型: {source_model}")
    print(f"  🎯 目标模型: {target_model}")
    print(f"  📋 问题数量: {max_problems}")
    print(f"  🔍 Beam数量: {num_beams}")
    print(f"  🔄 迭代次数: {n_iterations}")
    print(f"  📦 批次大小: {batch_size}")
    print(f"  🎲 采样方法: {args.sampling_method} (α={args.sampling_alpha})")
    print(f"  ⚖️ P2Value权重: {args.p2value_alpha}")
    print(f"  📁 输出目录: {output_dir}")
    print("=" * 80)
    
    if args.dry_run:
        print("🔍 Dry run模式，不执行实验")
        return
    
    # 询问确认
    response = input("是否继续执行实验？(y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ 实验已取消")
        return
    
    # LoRA配置
    lora_config = {
        'r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.05,
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'bias': 'none',
        'task_type': 'CAUSAL_LM'
    }
    
    # 创建训练参数
    training_args = create_optimized_training_args(
        f"{output_dir}/checkpoints", 
        args.experiment_size
    )
    
    # 创建实验
    print("\n🔄 初始化实验...")
    experiment = LocalBTPFineTuneExperiment(
        source_model_path=source_model,
        target_model_path=target_model,
        dataset="mbpp",
        sampling_method=args.sampling_method,
        sampling_alpha=args.sampling_alpha,
        p2value_alpha=args.p2value_alpha,
        use_lora=True,
        lora_config=lora_config
    )
    
    # 运行实验
    print("\n🚀 开始实验...")
    results = experiment.run_experiment(
        max_problems=max_problems,
        num_beams=num_beams,
        n_iterations=n_iterations,
        batch_size=batch_size,
        output_dir=output_dir
    )
    
    # 显示结果摘要
    print("\n" + "=" * 80)
    print("📊 实验结果摘要")
    print("=" * 80)
    final_stats = results['final_stats']
    initial_stats = results['initial_stats']
    
    print(f"📈 经验数据:")
    print(f"  总经验数: {final_stats.get('total_experiences', 0)}")
    print(f"  完美解决方案: {final_stats.get('perfect_solutions', 0)}")
    print(f"  平均通过率: {final_stats.get('avg_pass_rate', 0):.3f}")
    print(f"  平均P2Value: {final_stats.get('avg_p2value', 0):.3f}")
    
    if initial_stats.get('total_experiences', 0) > 0:
        improvement = (final_stats.get('avg_pass_rate', 0) - initial_stats.get('avg_pass_rate', 0))
        print(f"  通过率提升: {improvement:.3f}")
    
    print(f"\n📁 详细结果保存在: {output_dir}")
    print("🎉 实验完成!")


if __name__ == "__main__":
    main() 