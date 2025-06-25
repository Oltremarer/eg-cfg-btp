#!/usr/bin/env python3
"""
大模型采样 -> 小模型微调的BTP实验
实验思路：用DeepSeek-V3-0324采样，微调deepseek-coder-1.3b-instruct
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from step2_btp_finetune_experiment import BTPFineTuneExperiment
from transformers import TrainingArguments

def create_experiment_config():
    """创建实验配置"""
    return {
        # 模型配置 - 关键：大模型采样，小模型微调
        "source_model": "deepseek-ai/DeepSeek-V3-0324",  # 大模型用于采样
        "target_model": "deepseek-ai/deepseek-coder-1.3b-instruct",  # 小模型用于微调
        
        # BTP参数
        "num_beams": 8,  # 增加beam数量获得更多样的候选
        "n_iterations": 3,
        "batch_size": 64,
        "max_problems": 100,
        
        # 优先采样参数
        "sampling_method": "power",
        "sampling_alpha": 1.5,  # 适中的集中度
        "p2value_alpha": 0.7,   # 更重视通过率
        
        # LoRA配置 - 针对小模型优化
        "lora_config": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"  # 包含MLP层
            ]
        }
    }

def create_training_args(output_dir: str):
    """创建训练参数"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # 有效batch_size = 2 * 8 = 16
        learning_rate=1e-4,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=None  # 不上传到wandb等
    )

def run_big_to_small_experiment():
    """运行大模型采样->小模型微调实验"""
    
    # 创建配置
    config = create_experiment_config()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiments/results_archive/big_to_small_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    with open(f"{output_dir}/experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("=" * 60)
    print("🚀 大模型采样 -> 小模型微调 BTP实验")
    print("=" * 60)
    print(f"📊 采样模型: {config['source_model']}")
    print(f"🎯 目标模型: {config['target_model']}")
    print(f"🔄 微调迭代: {config['n_iterations']}")
    print(f"📦 批次大小: {config['batch_size']}")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)
    
    try:
        # 创建BTP实验
        experiment = BTPFineTuneExperiment(
            source_model_path=config['source_model'],
            target_model_path=config['target_model'],
            dataset="mbpp",
            sampling_method=config['sampling_method'],
            sampling_alpha=config['sampling_alpha'],
            p2value_alpha=config['p2value_alpha'],
            use_lora=True,
            lora_config=config['lora_config']
        )
        
        # 创建训练参数
        training_args = create_training_args(f"{output_dir}/checkpoints")
        
        # 运行实验
        results = experiment.run_experiment(
            max_problems=config['max_problems'],
            num_beams=config['num_beams'],
            n_iterations=config['n_iterations'],
            batch_size=config['batch_size'],
            training_args=training_args,
            output_dir=output_dir
        )
        
        # 保存结果
        with open(f"{output_dir}/experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ 实验完成！")
        print(f"📊 结果保存在: {output_dir}")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="大模型采样->小模型微调BTP实验")
    parser.add_argument("--max_problems", type=int, default=100, help="最大问题数量")
    parser.add_argument("--num_beams", type=int, default=8, help="Beam Search数量")
    parser.add_argument("--n_iterations", type=int, default=3, help="微调迭代次数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    
    args = parser.parse_args()
    
    # 更新配置
    config = create_experiment_config()
    config.update({
        "max_problems": args.max_problems,
        "num_beams": args.num_beams,
        "n_iterations": args.n_iterations,
        "batch_size": args.batch_size
    })
    
    run_big_to_small_experiment()

if __name__ == "__main__":
    main() 