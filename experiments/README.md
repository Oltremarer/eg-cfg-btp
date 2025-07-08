# EG-CFG 实验框架 

## 📁 新的文件结构

实验现在按数据集进行组织，每个数据集都有独立的实验文件：

```
experiments/
├── mbpp/                           # MBPP数据集实验
│   ├── step1_baseline_experiment.py
│   ├── step2_btp_finetune_experiment.py  
│   ├── step3_ablation_study.py
│   ├── step4_hyperparameter_study.py
│   └── results/                    # MBPP实验结果
├── humaneval/                      # HumanEval数据集实验
│   ├── step1_baseline_experiment.py
│   ├── step2_btp_finetune_experiment.py
│   ├── step3_ablation_study.py
│   ├── step4_hyperparameter_study.py
│   └── results/                    # HumanEval实验结果
├── apps/                           # APPS数据集实验
│   ├── step1_baseline_experiment.py
│   ├── step2_btp_finetune_experiment.py
│   ├── step3_ablation_study.py
│   ├── step4_hyperparameter_study.py
│   └── results/                    # APPS实验结果
├── shared/                         # 共享组件
│   ├── base_experiment.py          # 基础实验类
│   ├── dataset_configs.py          # 数据集配置
│   └── common_utils.py             # 通用工具函数
└── README.md                       # 本文档
```

## 🧩 共享组件

### 基础实验类 (`shared/base_experiment.py`)

提供了三个核心基类：

- **`BaseExperiment`**: 所有实验的基础抽象类
- **`DatasetExperiment`**: 数据集特定实验的基类  
- **`Step1BaselineExperiment`**: Step1基线实验基类
- **`Step2BTPExperiment`**: Step2 BTP实验基类

### 数据集配置 (`shared/dataset_configs.py`)

为每个数据集提供：
- 数据集特定参数（超时时间、最大代码长度等）
- 英文提示模板（避免中文生成问题）
- 默认实验参数
- 评估指标定义

### 通用工具 (`shared/common_utils.py`)

提供常用功能：
- 数据集加载器
- 安全代码执行
- 结果格式化
- 错误日志记录
- 进度显示

## 🎯 实验步骤说明

### Step 1: 基线性能测试
测试模型在各数据集上的原始性能，为后续对比提供基准。

**MBPP示例**：
```bash
cd experiments/mbpp
python step1_baseline_experiment.py \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --num_samples 10 \
    --max_problems 50
```

### Step 2: BTP微调实验  
使用Beam Search + Testing + Prioritized Experience Replay进行模型增强。

**特性**：
- 束搜索生成多个候选解决方案
- 自动测试验证解决方案正确性
- 基于P2Value的优先经验回放
- 支持LoRA微调

### Step 3: 消融研究
系统性地移除BTP的各个组件，验证每个部分的贡献。

### Step 4: 超参数研究
探索关键超参数对性能的影响，找到最优配置。

## 🔧 使用方法

### 1. 选择数据集
进入对应的数据集文件夹：
```bash
cd experiments/mbpp          # 或 humaneval, apps
```

### 2. 运行实验
每个实验文件都是独立的，可直接运行：
```bash
python step1_baseline_experiment.py --help  # 查看参数说明
```

### 3. 查看结果
实验结果自动保存到对应的 `results/` 文件夹中。

## 📊 支持的数据集

### MBPP (Mostly Basic Python Problems)
- **问题数量**: ~500个基础Python编程问题
- **难度**: 入门到中等
- **特点**: 清晰的问题描述和测试用例

### HumanEval  
- **问题数量**: 164个代码补全问题
- **难度**: 中等
- **特点**: 函数补全任务，真实编程场景

### APPS (Automated Programming Progress Standard)
- **问题数量**: ~10,000个编程问题
- **难度**: 入门、面试、竞赛三个级别
- **特点**: 大规模、多难度级别

## ⚙️ 配置说明

### 模型支持
- **本地模型**: DeepSeek、CodeLlama、StarCoder等
- **API模型**: OpenAI GPT系列（需要API密钥）
- **推理端点**: 支持自定义推理服务

### 实验参数
- `--model_name`: 模型名称或路径
- `--num_samples`: 每个问题的生成样本数
- `--temperature`: 生成温度（控制随机性）
- `--max_problems`: 最大测试问题数
- `--use_openai`: 使用OpenAI API

## 🏃‍♂️ 快速开始

### 1. 运行MBPP基线实验
```bash
cd experiments/mbpp
python step1_baseline_experiment.py \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --num_samples 2 \
    --max_problems 3
```

### 2. 运行BTP增强实验
```bash
python step2_btp_finetune_experiment.py \
    --source_model "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" \
    --target_model "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --max_problems 10
```

## 📈 结果分析

每个实验都会生成详细的结果文件，包含：

- **Pass@k指标**: k个尝试中至少一次成功的概率
- **成功率**: 完全解决问题的比例  
- **详细日志**: 每个问题的生成和测试过程
- **错误分析**: 失败原因和调试信息

## 🔍 故障排除

### 常见问题

1. **中文乱码问题**: 新框架默认使用英文提示，避免了中文导致的生成问题
2. **模型加载失败**: 检查模型路径和权限，确保有足够的GPU内存
3. **测试用例失败**: 查看详细错误日志，检查代码语法和逻辑

### 调试建议

- 使用小样本数测试（`--max_problems 3 --num_samples 2`）
- 查看生成的代码片段了解模型行为
- 检查实验配置是否与数据集匹配

---

## 🎉 优势

新的组织结构带来以下好处：

1. **清晰分离**: 每个数据集独立，避免混淆
2. **代码复用**: 共享基础类减少重复代码
3. **易于扩展**: 添加新数据集只需创建新文件夹
4. **标准化**: 统一的接口和配置格式
5. **英文提示**: 避免中文导致的模型生成问题

原始的通用实验文件仍在根目录保留，可作为参考。 