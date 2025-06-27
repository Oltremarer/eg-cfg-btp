# EG-CFG 实验套件

基于论文《Enhancing LLMs for Code Generation with Possibility and Pass rate Prioritized Experience Replay》的完整实验实现。

## 🎯 **完整实验计划与设计**

### 📋 **（一）实验目标与假设**
- **核心假设**：EG-CFG的Beam Search + Testing + Prioritized Experience Replay (BTP)框架能显著提升代码生成性能
- **验证目标**：
  1. EG-CFG相比baseline方法的性能提升 
  2. BTP框架各组件的有效性和贡献度
  3. 不同模型规模和类型上的通用性
  4. 超参数敏感性和最优配置

### 🔧 **（二）基础模型与Baseline选择**

#### **必选Baseline模型**（论文标准对比）：
```bash
# 传统基础模型
- GPT-2 / GPT-Neo-2.7B          # 轻量级基准
- CodeLlama-34B                 # 大型开源代码模型
- StarCoder2-15B                # 最新开源代码模型
- WizardCoder-34B               # 指令微调代码模型
```

#### **推荐对比模型**（最新SOTA）：
```bash
# DeepSeek系列
- DeepSeek-Coder-1.3B/6.7B/33B # 我们的主要测试模型
- DeepSeek-V3-0324              # 最新SOTA模型

# 云端模型
- GPT-4o / GPT-3.5-turbo        # OpenAI SOTA
- MathCoder2-7B                 # 数学增强版

# 轻量级模型
- SmolLM2系列 (135M/360M/1.7B)  # 资源受限场景
```

### 📊 **（三）实验数据集及评估指标**

#### **数据集**（建议通用广泛任务类型）：
```bash
✅ HumanEval    # 标准函数级任务 (164题)
✅ MBPP         # 基础编程问题 (500题) 
🔄 APPS         # 不同难度级别 (待集成)
🔄 CodeContests # 高难度竞赛题 (待集成)
```

#### **评估指标**：
```bash
🎯 Pass@1 / Pass@k     # 最重要指标 - 代码正确性
📊 Exact Match Accuracy # 精确匹配准确率
⏱️ 运行效率           # 时间和资源消耗
💡 P2Value分数        # EG-CFG特有指标
🔄 RSR (Relative Success Rate) # 相对成功率提升
```

---

## 🚀 最新更新：统一实验脚本

我们已经将所有step2相关的实验整合到一个统一的脚本中：`step2_btp_finetune_experiment.py`

### 🔥 统一脚本特性

**支持的实验模式：**
1. **本地模型BTP实验**（不含微调）- `--mode btp_only`
2. **本地模型微调** - `--mode finetune`
3. **OpenAI API实验** - `--mode openai`
4. **DeepSeek API实验** - `--mode deepseek`
5. **混合模式**（API采样+本地微调）- `--mode hybrid`

### 📝 使用示例

#### 1. 本地模型BTP实验（无微调）
```bash
python experiments/step2_btp_finetune_experiment.py \
  --source-model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode btp_only \
  --max-problems 50 \
  --num-beams 5
```

#### 2. 本地模型微调实验
```bash
python experiments/step2_btp_finetune_experiment.py \
  --source-model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --target-model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune \
  --max-problems 100 \
  --n-iterations 3 \
  --sampling-method power \
  --sampling-alpha 1.5
```

#### 3. OpenAI BTP实验
```bash
python experiments/step2_btp_finetune_experiment.py \
  --source-model gpt-4 \
  --mode openai \
  --api-key YOUR_OPENAI_API_KEY \
  --max-problems 30 \
  --num-beams 5
```

#### 4. DeepSeek API实验
```bash
python experiments/step2_btp_finetune_experiment.py \
  --source-model deepseek-chat \
  --mode deepseek \
  --api-key YOUR_DEEPSEEK_API_KEY \
  --api-base https://api.deepseek.com \
  --max-problems 30
```

#### 5. 混合模式（API采样+本地微调）
```bash
python experiments/step2_btp_finetune_experiment.py \
  --source-model gpt-4 \
  --target-model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode hybrid \
  --api-key YOUR_OPENAI_API_KEY \
  --max-problems 50 \
  --n-iterations 2
```

### 🎛️ 主要参数说明

| 参数组 | 参数 | 说明 | 默认值 |
|--------|------|------|--------|
| **模式** | `--mode` | 实验模式：btp_only/finetune/openai/deepseek/hybrid | finetune |
| **模型** | `--source-model` | 源模型路径或名称 | 必需 |
| | `--target-model` | 目标模型路径（用于微调） | None |
| **API** | `--api-key` | API密钥 | None |
| | `--api-base` | API基础URL | None |
| **数据** | `--dataset` | 数据集：mbpp/humaneval | mbpp |
| | `--max-problems` | 最大问题数量 | 50 |
| **BTP** | `--num-beams` | Beam Search数量 | 5 |
| | `--n-iterations` | PPER训练迭代次数 | 2 |
| | `--batch-size` | 训练批大小 | 50 |
| **采样** | `--sampling-method` | 采样方法：power/rank | power |
| | `--sampling-alpha` | 采样α参数 | 1.0 |
| | `--p2value-alpha` | P2Value权重α | 0.5 |
| **LoRA** | `--use-lora` | 使用LoRA微调 | True |
| | `--lora-r` | LoRA rank | 16 |
| | `--lora-alpha` | LoRA alpha | 32 |

### 🔧 模型类型自动识别

统一脚本会根据模型名称和模式自动识别模型类型：

- **本地模型**：`deepseek-ai/xxx`、`HuggingFaceTB/xxx`、`codellama/xxx`等
- **OpenAI模型**：`gpt-4`、`gpt-4o`、`gpt-3.5-turbo`等，或者`--mode openai`
- **DeepSeek API**：`deepseek-chat`等，或者`--mode deepseek`

### 💡 推荐配置

**快速测试（小规模）：**
```bash
--max-problems 10 --num-beams 3 --n-iterations 1
```

**标准实验（中等规模）：**
```bash
--max-problems 50 --num-beams 5 --n-iterations 2
```

**完整评估（大规模）：**
```bash
--max-problems 100 --num-beams 8 --n-iterations 3
```

### 🚀 主实验运行器

同时，我们提供了主实验运行器 `main_experiment.py`，可以运行所有类型的实验：

```bash
# 运行完整实验套件
python experiments/main_experiment.py \
  --experiment all \
  --model deepseek-1.3b \
  --mode quick

# 运行单个BTP微调实验
python experiments/main_experiment.py \
  --experiment btp_finetune \
  --source-model deepseek-v2-lite \
  --target-model deepseek-1.3b \
  --max-problems 100
```

---

## 🧪 其他实验脚本

### 基线实验
评测标准的采样-过滤方法在代码生成任务上的性能。

```bash
python experiments/step1_baseline_experiment.py \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset mbpp \
    --num_samples 10 \
    --max_problems 50 \
    --output_dir experiments/results/baseline
```

### 消融研究
对比不同经验回放策略的效果，证明P2Value优先采样的优势。

```bash
python experiments/step3_ablation_study.py \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset mbpp \
    --num_problems 30 \
    --output_dir experiments/results/ablation
```

### 超参数分析
探索关键超参数对BTP性能的影响，找到最优配置。

```bash
python experiments/step4_hyperparameter_study.py \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset mbpp \
    --search_type focused \
    --max_configs 15 \
    --output_dir experiments/results/hyperparameter
```

### 大小模型实验
大模型采样 → 小模型微调的完整流程。

```bash
python experiments/big_to_small_finetune_experiment.py \
    --source-model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --target-model deepseek-ai/deepseek-coder-1.3b-instruct \
    --max-problems 100
```

---

## 📊 结果分析

### 输出文件结构

```
experiments/results/experiment_YYYYMMDD_HHMMSS/
├── step1_baseline/
│   └── baseline_experiment_results.json
├── step2_btp/
│   └── btp_results.json
├── step3_ablation/
│   └── ablation_results.json
├── step4_hyperparameter/
│   └── hyperparameter_study_results.json
└── experiment_report.txt
```

### 关键指标

1. **Pass@k**: k个生成解决方案中至少一个完全正确的概率
2. **Average Pass Rate**: 平均测试用例通过率
3. **P2Value**: 综合生成概率和通过率的优先级指标
4. **Success Rate**: 完全解决问题的比例

---

## 🔧 环境要求

### 必需依赖
```bash
pip install torch transformers datasets tqdm numpy
```

### 可选依赖（根据使用模式）
```bash
# OpenAI API支持
pip install openai

# DeepSeek/其他API支持
pip install requests

# LoRA微调支持
pip install peft

# 高级功能
pip install accelerate bitsandbytes
```

---

## 🎯 最佳实践

1. **开始小规模测试**：使用`--max-problems 10`验证配置
2. **选择合适的采样策略**：Power Sampling通常效果更好
3. **调整P2Value权重**：`--p2value-alpha 0.3`更重视通过率
4. **使用LoRA微调**：减少显存占用，提高训练效率
5. **监控实验进度**：开启`--debug`查看详细日志

---

## 🆘 故障排除

**常见问题：**
- **内存不足**：减少`--batch-size`和`--per-device-batch-size`
- **API调用失败**：检查`--api-key`和网络连接
- **模型加载失败**：确认模型路径和权限
- **CUDA错误**：检查GPU驱动和CUDA版本兼容性

**获取帮助：**
```bash
python experiments/step2_btp_finetune_experiment.py --help
``` 