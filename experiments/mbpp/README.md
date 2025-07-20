# MBPP BTP实验指南

## 概述

本实验实现了基于MBPP数据集的BTP（Beam Search + Testing + Prioritized Experience Replay）算法，支持多种工作流和缓存机制。

## 核心功能

### 1. 性能优化
- **学习率优化**: 已将学习率从1e-4降低到1e-6，防止灾难性遗忘
- **LoRA微调**: 支持高效的参数高效微调
- **智能缓存**: 支持采样结果缓存，避免重复计算

### 2. 两种缓存机制

#### 工作流A：自动化经验池缓存（推荐）
- **目标**: 一次性生成经验池，后续快速实验
- **优势**: 大幅提升实验效率，支持快速超参数调优

#### 工作流B：固定训练样本
- **目标**: 确保不同实验使用完全一致的训练数据
- **优势**: 保证实验的公平性和可重现性

## 使用指南

### 工作流A：自动化经验池缓存

#### 步骤1：首次运行（生成缓存）
```bash
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 100 --num-beams 5
```

**说明**:
- 使用`finetune`模式
- 执行完整的阶段1（代码生成+测试）
- 自动保存采样结果到缓存文件
- 继续执行阶段2（微调训练）

#### 步骤2：后续运行（使用缓存）
```bash
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 100 --num-beams 5
```

**说明**:
- 检测到缓存文件存在
- 直接加载缓存数据，跳过阶段1
- 秒级进入阶段2微调
- 可以修改超参数进行快速实验

### 工作流B：固定训练样本

#### 步骤1：生成固定样本
```bash
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode local --max-problems 100 \
  --fixed-sample-path ./fixed_samples.json
```

**说明**:
- 使用`local`模式（不进行微调）
- 执行阶段1生成经验池
- 采样一批数据并保存到指定文件
- 结束运行（跳过阶段2）

#### 步骤2：使用固定样本微调
```bash
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 100 \
  --fixed-sample-path ./fixed_samples.json
```

**说明**:
- 加载指定的固定样本文件
- 使用完全一致的训练数据进行微调
- 确保不同实验间的公平对比

## 高级选项

### 强制重新采样
```bash
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 100 \
  --force-resample
```

### 禁用缓存功能
```bash
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 100 \
  --use-cached-sampling false
```

## 参数说明

### 基本参数
- `--model`: 模型名称或路径
- `--mode`: 实验模式（local/finetune/openai/deepseek）
- `--max-problems`: 最大问题数量
- `--num-beams`: Beam Search数量
- `--n-iterations`: PPER训练迭代次数
- `--batch-size`: 训练批大小

### 缓存参数
- `--use-cached-sampling`: 使用缓存的采样结果（默认True）
- `--force-resample`: 强制重新采样，忽略缓存
- `--fixed-sample-path`: 固定样本文件路径

### BTP算法参数
- `--sampling-method`: 采样方法（power/rank）
- `--sampling-alpha`: 采样α参数
- `--p2value-alpha`: P2Value权重α

## 输出文件

### 缓存文件位置
- 自动缓存: `{output_dir}/sampling_cache/`
- 固定样本: 用户指定的路径

### 模型保存
- 微调模型: `{output_dir}/`目录
- 实验结果: `experiments/mbpp/results/`

## 实验建议

### 1. 首次实验
建议使用工作流A，先用较小的数据集（如50个问题）进行测试：
```bash
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 50 --num-beams 5
```

### 2. 大规模实验
确认流程无误后，可以扩展到更大的数据集：
```bash
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 500 --num-beams 10
```

### 3. 多GPU并行实验
对于大规模数据集，可以使用多GPU并行处理：

#### GPU 0 - DeepSeek模型
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode local --max-problems 974 --num-beams 10 \
  --output-dir ./btp_workspace_deepseek \
  --gpu-id 0 --save-interval 50 --force-resample
```

#### GPU 1 - CodeLlama模型
```bash
CUDA_VISIBLE_DEVICES=1 python experiments/mbpp/step2_btp_experiment.py \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --mode local --max-problems 974 --num-beams 10 \
  --output-dir ./btp_workspace_codellama \
  --gpu-id 0 --save-interval 50 --force-resample
```

**说明**:
- `CUDA_VISIBLE_DEVICES=X`: 指定使用的GPU
- `--gpu-id 0`: 在指定GPU上使用设备0
- `--save-interval 50`: 每50个问题保存一次进度
- `--force-resample`: 强制重新采样
- 进度文件保存在各自的输出目录中

### 3. 对比实验
使用工作流B确保实验的公平性：
```bash
# 生成固定样本
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode local --max-problems 100 \
  --fixed-sample-path ./experiment_samples.json

# 使用固定样本进行不同超参数的实验
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 100 \
  --fixed-sample-path ./experiment_samples.json \
  --sampling-alpha 0.5

python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 100 \
  --fixed-sample-path ./experiment_samples.json \
  --sampling-alpha 1.0
```

## 故障排除

### 常见问题

1. **缓存加载失败**
   - 检查缓存文件是否存在
   - 确认模型名称和参数匹配
   - 使用`--force-resample`重新生成

2. **内存不足**
   - 减少`--max-problems`数量
   - 减少`--num-beams`数量
   - 减少`--batch-size`大小

3. **训练失败**
   - 检查学习率设置
   - 确认数据集格式正确
   - 查看详细错误日志

### 调试模式
```bash
python experiments/mbpp/step2_btp_experiment.py \
  --model deepseek-ai/deepseek-coder-1.3b-instruct \
  --mode finetune --max-problems 10 \
  --debug
```

## 技术细节

### BTP算法流程
1. **阶段1**: Beam Search采样 + 测试验证
2. **阶段2**: 优先经验回放（PPER）训练

### 缓存机制
- **自动缓存**: 基于模型名称、问题数量、beam数量自动命名
- **固定样本**: 用户指定路径，确保实验一致性

### 性能优化
- **学习率**: 1e-6（防止灾难性遗忘）
- **LoRA**: 参数高效微调
- **缓存**: 避免重复采样
- **批处理**: 支持大规模数据处理 