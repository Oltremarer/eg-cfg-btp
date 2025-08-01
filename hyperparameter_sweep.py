import os
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 1. 在这里定义你想要测试的超参数组合
# ==============================================================================
# 每一组{}代表一次独立的实验。你可以随意增删或修改。
# "run_name" 会被用作输出目录名，所以请保持其唯一性。
PARAMETER_GRID = [
    {
        "run_name": "lr_1e-5_r_8_epoch_3",
        "learning_rate": 1e-5,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_train_epochs": 3,
    },
    {
        "run_name": "lr_2e-5_r_8_epoch_3",
        "learning_rate": 2e-5,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_train_epochs": 3,
    },
    {
        "run_name": "lr_1e-5_r_16_epoch_3",
        "learning_rate": 1e-5,
        "lora_r": 16,
        "lora_alpha": 32,
        "num_train_epochs": 3,
    },
    {
        "run_name": "lr_2e-5_r_16_epoch_5_epochs",
        "learning_rate": 2e-5,
        "lora_r": 16,
        "lora_alpha": 32,
        "num_train_epochs": 5, # 尝试更多训练周期
    },
]

# ==============================================================================
# 2. 在这里配置你的通用参数
# ==============================================================================
# 这是所有实验共享的参数，请根据你的实际情况修改。
BASE_CONFIG = {
    "script_path": "experiments/mbpp/step2_btp_experiment.py",
    "base_model": "deepseek-ai/deepseek-coder-1.3b-instruct",
    # 【非常重要】请在这里填写你用V2生成的缓存数据路径！
    "sample_cache_path": "./results/Deepseek_V2_mbpp_all/sampling_cache/sampling_cache_deepseek_ai_DeepSeek_Coder_V2_Lite_Instruct_max974_beams5.json",
    "sweep_output_base_dir": "./sweep_results", # 所有实验结果的总目录
    # 评估时使用的题目数量
    "eval_max_problems": 50,
}

def run_command(command: list):
    """辅助函数：运行命令行指令并实时打印输出"""
    print("\n" + "="*80)
    print(f"🚀 EXECUTING: {' '.join(command)}")
    print("="*80 + "\n")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = process.poll()
        if rc != 0:
            print(f"❌ Command failed with return code {rc}")
        return rc
    except Exception as e:
        print(f"❌ An exception occurred while running command: {e}")
        return -1


def parse_pass_at_1(eval_output_dir: str) -> float:
    """
    辅助函数：从评估结果的JSON文件中解析出pass@1。
    注意：这里的实现基于step2脚本最终会统计'fully_passed_count'的假设。
    """
    try:
        # 查找评估目录下最新的json结果文件
        result_files = sorted(Path(eval_output_dir).glob('*.json'), key=os.path.getmtime, reverse=True)
        if not result_files:
            print(f"⚠️ No result JSON file found in {eval_output_dir}")
            return -1.0

        latest_result_file = result_files[0]
        with open(latest_result_file, 'r') as f:
            results = json.load(f)

        # 从结果中提取通过率
        total_experiences = results.get('experience_stats', {}).get('total_experiences', 0)
        # 这里的 total_experiences 实际上是评估的问题数
        total_problems_evaluated = BASE_CONFIG['eval_max_problems']

        # 统计pass@1，即beam search的任一beam通过就算通过
        passed_problems = set()
        for exp in results.get('all_experiences', []):
            if exp.get('pass_rate', 0) > 0:
                passed_problems.add(exp['problem_id'])

        if total_problems_evaluated == 0:
            return 0.0

        pass_at_1 = len(passed_problems) / total_problems_evaluated
        return pass_at_1 * 100 # 返回百分比
    except Exception as e:
        print(f"Error parsing pass@1 from {eval_output_dir}: {e}")
        return -1.0


def main():
    """主函数，编排所有实验"""
    
    # 确保step2脚本中已经加入了 save_strategy="epoch"
    print("🔔 Reminder: Please ensure that `save_strategy=\"epoch\"` is set in the TrainingArguments of your step2 script.")

    # 创建本次sweep的总目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(BASE_CONFIG["sweep_output_base_dir"]) / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 All results for this sweep will be saved in: {sweep_dir}")

    summary_results = []

    for i, params in enumerate(PARAMETER_GRID):
        run_name = params["run_name"]
        print("\n" + "#"*100)
        print(f"##  Starting Run {i+1}/{len(PARAMETER_GRID)}: {run_name}")
        print("#"*100 + "\n")

        run_output_dir = sweep_dir / run_name
        run_output_dir.mkdir(exist_ok=True)

        # --- 1. 训练 ---
        train_command = [
            "python", BASE_CONFIG["script_path"],
            "--model", BASE_CONFIG["base_model"],
            "--mode", "finetune",
            "--sample-cache-path", BASE_CONFIG["sample_cache_path"],
            "--output-dir", str(run_output_dir),
            "--learning-rate", str(params["learning_rate"]),
            "--lora-r", str(params["lora_r"]),
            "--lora-alpha", str(params["lora_alpha"]),
            "--num-train-epochs", str(params["num_train_epochs"]),
            # 使用一些合理的默认值
            "--per-device-batch-size", "4",
            "--grad-accum-steps", "8",
        ]
        
        run_command(train_command)

        # --- 2. 评估所有checkpoints ---
        best_pass_rate = -1.0
        best_checkpoint = "None"
        
        checkpoint_dirs = sorted([d for d in run_output_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint')])

        if not checkpoint_dirs:
            print(f"⚠️ No checkpoints found for run {run_name}. Skipping evaluation.")
            continue

        for checkpoint_dir in checkpoint_dirs:
            print(f"\n--- Evaluating checkpoint: {checkpoint_dir.name} ---")
            eval_output_dir = checkpoint_dir / "evaluation"
            eval_output_dir.mkdir(exist_ok=True)

            eval_command = [
                "python", BASE_CONFIG["script_path"],
                "--model", str(checkpoint_dir),
                "--mode", "local",
                "--max-problems", str(BASE_CONFIG["eval_max_problems"]),
                "--output-dir", str(eval_output_dir),
                "--force-resample", # 确保每次都重新评估而不是用缓存
            ]

            run_command(eval_command)
            
            pass_rate = parse_pass_at_1(str(eval_output_dir))
            print(f"✅ Pass@1 for {checkpoint_dir.name}: {pass_rate:.2f}%")

            if pass_rate > best_pass_rate:
                best_pass_rate = pass_rate
                best_checkpoint = checkpoint_dir.name
        
        summary_results.append({
            **params,
            "best_checkpoint": best_checkpoint,
            "best_pass_at_1": f"{best_pass_rate:.2f}%"
        })

    # --- 3. 打印最终总结报告 ---
    print("\n\n" + "*"*100)
    print("##  HYPERPARAMETER SWEEP COMPLETE - FINAL REPORT")
    print("*"*100 + "\n")
    
    # 打印表头
    headers = list(PARAMETER_GRID[0].keys()) + ["best_checkpoint", "best_pass_at_1"]
    header_line = " | ".join(f"{h:<25}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    # 打印每一行结果
    for result in summary_results:
        row_values = [str(result.get(h, 'N/A')) for h in headers]
        row_line = " | ".join(f"{v:<25}" for v in row_values)
        print(row_line)

if __name__ == "__main__":
    main() 