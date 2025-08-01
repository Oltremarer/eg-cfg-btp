import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# ==============================================================================
# 1. GPU 1 的参数组合 (共32组)
# ==============================================================================
PARAMETER_GRID = [
    # --- Learning Rate: 1e-4 ---
    {'run_name': 'lr_1e-4_r_4_e_3', 'learning_rate': 0.0001, 'lora_r': 4, 'lora_alpha': 8, 'num_train_epochs': 3},
    {'run_name': 'lr_1e-4_r_8_e_3', 'learning_rate': 0.0001, 'lora_r': 8, 'lora_alpha': 16, 'num_train_epochs': 3},
    {'run_name': 'lr_1e-4_r_16_e_3', 'learning_rate': 0.0001, 'lora_r': 16, 'lora_alpha': 32, 'num_train_epochs': 3},
    {'run_name': 'lr_1e-4_r_32_e_3', 'learning_rate': 0.0001, 'lora_r': 32, 'lora_alpha': 64, 'num_train_epochs': 3},
    {'run_name': 'lr_1e-4_r_4_e_5', 'learning_rate': 0.0001, 'lora_r': 4, 'lora_alpha': 8, 'num_train_epochs': 5},
    {'run_name': 'lr_1e-4_r_8_e_5', 'learning_rate': 0.0001, 'lora_r': 8, 'lora_alpha': 16, 'num_train_epochs': 5},
    {'run_name': 'lr_1e-4_r_16_e_5', 'learning_rate': 0.0001, 'lora_r': 16, 'lora_alpha': 32, 'num_train_epochs': 5},
    {'run_name': 'lr_1e-4_r_32_e_5', 'learning_rate': 0.0001, 'lora_r': 32, 'lora_alpha': 64, 'num_train_epochs': 5},
    # --- Learning Rate: 3e-4 ---
    {'run_name': 'lr_3e-4_r_4_e_3', 'learning_rate': 0.0003, 'lora_r': 4, 'lora_alpha': 8, 'num_train_epochs': 3},
    {'run_name': 'lr_3e-4_r_8_e_3', 'learning_rate': 0.0003, 'lora_r': 8, 'lora_alpha': 16, 'num_train_epochs': 3},
    {'run_name': 'lr_3e-4_r_16_e_3', 'learning_rate': 0.0003, 'lora_r': 16, 'lora_alpha': 32, 'num_train_epochs': 3},
    {'run_name': 'lr_3e-4_r_32_e_3', 'learning_rate': 0.0003, 'lora_r': 32, 'lora_alpha': 64, 'num_train_epochs': 3},
    {'run_name': 'lr_3e-4_r_4_e_5', 'learning_rate': 0.0003, 'lora_r': 4, 'lora_alpha': 8, 'num_train_epochs': 5},
    {'run_name': 'lr_3e-4_r_8_e_5', 'learning_rate': 0.0003, 'lora_r': 8, 'lora_alpha': 16, 'num_train_epochs': 5},
    {'run_name': 'lr_3e-4_r_16_e_5', 'learning_rate': 0.0003, 'lora_r': 16, 'lora_alpha': 32, 'num_train_epochs': 5},
    {'run_name': 'lr_3e-4_r_32_e_5', 'learning_rate': 0.0003, 'lora_r': 32, 'lora_alpha': 64, 'num_train_epochs': 5},
    # --- Learning Rate: 5e-4 ---
    {'run_name': 'lr_5e-4_r_4_e_3', 'learning_rate': 0.0005, 'lora_r': 4, 'lora_alpha': 8, 'num_train_epochs': 3},
    {'run_name': 'lr_5e-4_r_8_e_3', 'learning_rate': 0.0005, 'lora_r': 8, 'lora_alpha': 16, 'num_train_epochs': 3},
    {'run_name': 'lr_5e-4_r_16_e_3', 'learning_rate': 0.0005, 'lora_r': 16, 'lora_alpha': 32, 'num_train_epochs': 3},
    {'run_name': 'lr_5e-4_r_32_e_3', 'learning_rate': 0.0005, 'lora_r': 32, 'lora_alpha': 64, 'num_train_epochs': 3},
    {'run_name': 'lr_5e-4_r_4_e_5', 'learning_rate': 0.0005, 'lora_r': 4, 'lora_alpha': 8, 'num_train_epochs': 5},
    {'run_name': 'lr_5e-4_r_8_e_5', 'learning_rate': 0.0005, 'lora_r': 8, 'lora_alpha': 16, 'num_train_epochs': 5},
    {'run_name': 'lr_5e-4_r_16_e_5', 'learning_rate': 0.0005, 'lora_r': 16, 'lora_alpha': 32, 'num_train_epochs': 5},
    {'run_name': 'lr_5e-4_r_32_e_5', 'learning_rate': 0.0005, 'lora_r': 32, 'lora_alpha': 64, 'num_train_epochs': 5},
    # --- Learning Rate: 8e-4 ---
    {'run_name': 'lr_8e-4_r_4_e_3', 'learning_rate': 0.0008, 'lora_r': 4, 'lora_alpha': 8, 'num_train_epochs': 3},
    {'run_name': 'lr_8e-4_r_8_e_3', 'learning_rate': 0.0008, 'lora_r': 8, 'lora_alpha': 16, 'num_train_epochs': 3},
    {'run_name': 'lr_8e-4_r_16_e_3', 'learning_rate': 0.0008, 'lora_r': 16, 'lora_alpha': 32, 'num_train_epochs': 3},
    {'run_name': 'lr_8e-4_r_32_e_3', 'learning_rate': 0.0008, 'lora_r': 32, 'lora_alpha': 64, 'num_train_epochs': 3},
    {'run_name': 'lr_8e-4_r_4_e_5', 'learning_rate': 0.0008, 'lora_r': 4, 'lora_alpha': 8, 'num_train_epochs': 5},
    {'run_name': 'lr_8e-4_r_8_e_5', 'learning_rate': 0.0008, 'lora_r': 8, 'lora_alpha': 16, 'num_train_epochs': 5},
    {'run_name': 'lr_8e-4_r_16_e_5', 'learning_rate': 0.0008, 'lora_r': 16, 'lora_alpha': 32, 'num_train_epochs': 5},
    {'run_name': 'lr_8e-4_r_32_e_5', 'learning_rate': 0.0008, 'lora_r': 32, 'lora_alpha': 64, 'num_train_epochs': 5},
]

# ==============================================================================
# 2. 通用参数配置
# ==============================================================================
BASE_CONFIG = {
    "script_path": "experiments/mbpp/step2_btp_experiment.py",
    "base_model": "deepseek-ai/deepseek-coder-1.3b-instruct",
    # 【已更新】这是你提供的采样数据路径
    "sample_cache_path": "./results/Deepseek_V2_mbpp_all/sampling_cache/sampling_cache_deepseek_ai_DeepSeek_Coder_V2_Lite_Instruct_max974_beams5.json",
    "sweep_output_base_dir": "./sweep_results",
    # 【已更新】评估时使用的问题数量
    "eval_max_problems": 10,
    # 【已更新】数据采样批次大小
    "train_batch_size": 300,
    # 【已更新】指定此脚本使用的GPU
    "gpu_id": 1,
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
    """辅助函数：从评估结果的JSON文件中解析出pass@1"""
    try:
        result_files = sorted(Path(eval_output_dir).glob('*.json'), key=os.path.getmtime, reverse=True)
        if not result_files:
            return -1.0
        latest_result_file = result_files[0]
        with open(latest_result_file, 'r') as f:
            results = json.load(f)
        total_problems_evaluated = BASE_CONFIG['eval_max_problems']
        passed_problems = set()
        for exp in results.get('all_experiences', []):
            if exp.get('pass_rate', 0) > 0:
                passed_problems.add(exp['problem_id'])
        if total_problems_evaluated == 0:
            return 0.0
        return (len(passed_problems) / total_problems_evaluated) * 100
    except Exception as e:
        print(f"Error parsing pass@1 from {eval_output_dir}: {e}")
        return -1.0


def main():
    """主函数，编排所有实验"""
    print(f"🔔 Starting sweep on GPU {BASE_CONFIG['gpu_id']}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(BASE_CONFIG["sweep_output_base_dir"]) / f"sweep_gpu{BASE_CONFIG['gpu_id']}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 All results for this sweep will be saved in: {sweep_dir}")
    summary_results = []
    for i, params in enumerate(PARAMETER_GRID):
        run_name = params["run_name"]
        print("\n" + "#"*100)
        print(f"##  Starting Run {i+1}/{len(PARAMETER_GRID)}: {run_name} on GPU {BASE_CONFIG['gpu_id']}")
        print("#"*100 + "\n")
        run_output_dir = sweep_dir / run_name
        run_output_dir.mkdir(exist_ok=True)
        train_command = [
            "python", BASE_CONFIG["script_path"],
            "--model", BASE_CONFIG["base_model"],
            "--mode", "finetune",
            "--sample-cache-path", BASE_CONFIG["sample_cache_path"],
            "--output-dir", str(run_output_dir),
            "--batch-size", str(BASE_CONFIG["train_batch_size"]),
            "--gpu-id", str(BASE_CONFIG["gpu_id"]),
            "--learning-rate", str(params["learning_rate"]),
            "--lora-r", str(params["lora_r"]),
            "--lora-alpha", str(params["lora_alpha"]),
            "--num-train-epochs", str(params["num_train_epochs"]),
            "--per-device-batch-size", "4",
            "--grad-accum-steps", "8",
        ]
        run_command(train_command)
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
                "--gpu-id", str(BASE_CONFIG["gpu_id"]),
                "--force-resample",
            ]
            run_command(eval_command)
            pass_rate = parse_pass_at_1(str(eval_output_dir))
            print(f"✅ Pass@1 for {checkpoint_dir.name}: {pass_rate:.2f}%")
            if pass_rate > best_pass_rate:
                best_pass_rate = pass_rate
                best_checkpoint = checkpoint_dir.name
        summary_results.append({**params, "best_checkpoint": best_checkpoint, "best_pass_at_1": f"{best_pass_rate:.2f}%"})
    
    print("\n\n" + "*"*100)
    print(f"##  GPU {BASE_CONFIG['gpu_id']} SWEEP COMPLETE - FINAL REPORT")
    print("*"*100 + "\n")
    headers = list(PARAMETER_GRID[0].keys()) + ["best_checkpoint", "best_pass_at_1"]
    header_line = " | ".join(f"{h:<25}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for result in summary_results:
        row_values = [str(result.get(h, 'N/A')) for h in headers]
        row_line = " | ".join(f"{v:<25}" for v in row_values)
        print(row_line)

if __name__ == "__main__":
    main() 