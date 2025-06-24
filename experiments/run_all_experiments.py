#!/usr/bin/env python3
"""
主实验脚本：运行完整的BTP实验流程
包括：基线、BTP、消融研究、超参数分析
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, model_name: str, dataset: str = "mbpp", base_output_dir: str = "experiments/results"):
        self.model_name = model_name
        self.dataset = dataset
        self.base_output_dir = Path(base_output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建实验目录
        self.experiment_dir = self.base_output_dir / f"experiment_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Model: {model_name}, Dataset: {dataset}")
    
    def run_step1_baseline(self, num_samples=10, max_problems=50):
        """运行步骤1：基线实验"""
        print("\n" + "="*50)
        print("STEP 1: BASELINE EXPERIMENT")
        print("="*50)
        
        output_dir = self.experiment_dir / "step1_baseline"
        
        cmd = [
            sys.executable, "experiments/step1_baseline_experiment.py",
            "--model_name", self.model_name,
            "--dataset", self.dataset,
            "--output_dir", str(output_dir),
            "--num_samples", str(num_samples),
            "--max_problems", str(max_problems)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Baseline experiment completed successfully")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print("❌ Baseline experiment failed")
            print(f"Error: {e.stderr}")
            return False
    
    def run_step2_btp(self, collect_problems=50):
        """运行步骤2：BTP实验"""
        print("\n" + "="*50)
        print("STEP 2: BTP EXPERIMENT")
        print("="*50)
        
        output_dir = self.experiment_dir / "step2_btp"
        
        cmd = [
            sys.executable, "experiments/step2_btp_experiment.py",
            "--model_name", self.model_name,
            "--dataset", self.dataset,
            "--output_dir", str(output_dir),
            "--collect_problems", str(collect_problems)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ BTP experiment completed successfully")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print("❌ BTP experiment failed")
            print(f"Error: {e.stderr}")
            return False
    
    def run_step3_ablation(self, num_problems=30):
        """运行步骤3：消融研究"""
        print("\n" + "="*50)
        print("STEP 3: ABLATION STUDY")
        print("="*50)
        
        output_dir = self.experiment_dir / "step3_ablation"
        
        cmd = [
            sys.executable, "experiments/step3_ablation_study.py",
            "--model_name", self.model_name,
            "--dataset", self.dataset,
            "--output_dir", str(output_dir),
            "--num_problems", str(num_problems)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Ablation study completed successfully")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print("❌ Ablation study failed")
            print(f"Error: {e.stderr}")
            return False
    
    def run_step4_hyperparameter(self, search_type="focused", max_configs=15):
        """运行步骤4：超参数分析"""
        print("\n" + "="*50)
        print("STEP 4: HYPERPARAMETER STUDY")
        print("="*50)
        
        output_dir = self.experiment_dir / "step4_hyperparameter"
        
        cmd = [
            sys.executable, "experiments/step4_hyperparameter_study.py",
            "--model_name", self.model_name,
            "--dataset", self.dataset,
            "--output_dir", str(output_dir),
            "--search_type", search_type,
            "--max_configs", str(max_configs)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Hyperparameter study completed successfully")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print("❌ Hyperparameter study failed")
            print(f"Error: {e.stderr}")
            return False
    
    def run_all_experiments(self, config=None):
        """运行所有实验"""
        if config is None:
            config = {
                'baseline': {'num_samples': 10, 'max_problems': 50},
                'btp': {'collect_problems': 50},
                'ablation': {'num_problems': 30},
                'hyperparameter': {'search_type': 'focused', 'max_configs': 15}
            }
        
        print(f"Starting complete experiment suite for {self.model_name}")
        print(f"Experiment configuration: {config}")
        
        results = {}
        
        # 步骤1：基线实验
        if 'baseline' in config:
            results['step1_baseline'] = self.run_step1_baseline(**config['baseline'])
        
        # 步骤2：BTP实验
        if 'btp' in config:
            results['step2_btp'] = self.run_step2_btp(**config['btp'])
        
        # 步骤3：消融研究
        if 'ablation' in config:
            results['step3_ablation'] = self.run_step3_ablation(**config['ablation'])
        
        # 步骤4：超参数分析
        if 'hyperparameter' in config:
            results['step4_hyperparameter'] = self.run_step4_hyperparameter(**config['hyperparameter'])
        
        # 生成实验报告
        self.generate_experiment_report(results)
        
        return results
    
    def generate_experiment_report(self, results):
        """生成实验报告"""
        print("\n" + "="*60)
        print("EXPERIMENT SUITE SUMMARY")
        print("="*60)
        
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset}")
        print(f"Experiment Directory: {self.experiment_dir}")
        print(f"Timestamp: {self.timestamp}")
        
        print("\nExperiment Results:")
        for step, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {step}: {status}")
        
        successful_steps = sum(results.values())
        total_steps = len(results)
        print(f"\nOverall Success Rate: {successful_steps}/{total_steps} ({successful_steps/total_steps*100:.1f}%)")
        
        # 保存报告到文件
        report_file = self.experiment_dir / "experiment_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"BTP Experiment Suite Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Experiment Directory: {self.experiment_dir}\n\n")
            f.write("Results:\n")
            for step, success in results.items():
                status = "SUCCESS" if success else "FAILED"
                f.write(f"  {step}: {status}\n")
            f.write(f"\nSuccess Rate: {successful_steps}/{total_steps}\n")
        
        print(f"\nDetailed report saved to: {report_file}")
        print("="*60)
    
    def run_quick_test(self):
        """运行快速测试（小规模实验）"""
        print("Running quick test with reduced parameters...")
        
        quick_config = {
            'baseline': {'num_samples': 3, 'max_problems': 10},
            'btp': {'collect_problems': 10},
            'ablation': {'num_problems': 10},
            'hyperparameter': {'search_type': 'focused', 'max_configs': 5}
        }
        
        return self.run_all_experiments(quick_config)


def main():
    parser = argparse.ArgumentParser(description="Run Complete BTP Experiment Suite")
    parser.add_argument("--model_name", type=str, required=True, 
                       help="Model name (e.g., 'deepseek-ai/deepseek-coder-1.3b-instruct')")
    parser.add_argument("--dataset", type=str, default="mbpp", 
                       choices=["mbpp", "humaneval"], help="Dataset to use")
    parser.add_argument("--output_dir", type=str, default="experiments/results", 
                       help="Base output directory")
    parser.add_argument("--mode", type=str, choices=["full", "quick", "single"], 
                       default="full", help="Experiment mode")
    parser.add_argument("--single_step", type=str, 
                       choices=["baseline", "btp", "ablation", "hyperparameter"],
                       help="Single step to run (when mode=single)")
    
    # 实验参数
    parser.add_argument("--num_samples", type=int, default=10, 
                       help="Number of samples for baseline")
    parser.add_argument("--max_problems", type=int, default=50, 
                       help="Maximum problems for baseline")
    parser.add_argument("--collect_problems", type=int, default=50, 
                       help="Problems to collect for BTP")
    parser.add_argument("--ablation_problems", type=int, default=30, 
                       help="Problems for ablation study")
    parser.add_argument("--max_configs", type=int, default=15, 
                       help="Maximum configs for hyperparameter study")
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = ExperimentRunner(args.model_name, args.dataset, args.output_dir)
    
    if args.mode == "quick":
        # 快速测试模式
        results = runner.run_quick_test()
    elif args.mode == "single":
        # 单步模式
        if not args.single_step:
            print("Error: --single_step required when mode=single")
            return
        
        if args.single_step == "baseline":
            results = {"baseline": runner.run_step1_baseline(args.num_samples, args.max_problems)}
        elif args.single_step == "btp":
            results = {"btp": runner.run_step2_btp(args.collect_problems)}
        elif args.single_step == "ablation":
            results = {"ablation": runner.run_step3_ablation(args.ablation_problems)}
        elif args.single_step == "hyperparameter":
            results = {"hyperparameter": runner.run_step4_hyperparameter("focused", args.max_configs)}
    
    else:
        # 完整实验模式
        config = {
            'baseline': {'num_samples': args.num_samples, 'max_problems': args.max_problems},
            'btp': {'collect_problems': args.collect_problems},
            'ablation': {'num_problems': args.ablation_problems},
            'hyperparameter': {'search_type': 'focused', 'max_configs': args.max_configs}
        }
        results = runner.run_all_experiments(config)
    
    print("\nExperiment suite completed!")


if __name__ == "__main__":
    main() 