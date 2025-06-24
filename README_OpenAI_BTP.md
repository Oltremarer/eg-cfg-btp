# OpenAI BTP实验使用指南

## 概述

本文档介绍如何使用OpenAI GPT API运行BTP（Beam Search + Testing + Prioritized Experience Replay）实验。

## 什么是BTP？

BTP是一个增强的代码生成方法，包含三个核心组件：

1. **Beam Search（束搜索）**: 生成多个候选解决方案
2. **Testing（测试）**: 对每个候选解决方案进行测试
3. **Prioritized Experience Replay（优先经验回放）**: 基于P2Value（可能性+通过率）对经验进行优先级排序和学习

## P2Value指标

P2Value = α × 可能性分数 + (1-α) × 测试通过率

其中：
- 可能性分数：基于生成温度和候选排名的启发式评分
- 测试通过率：解决方案通过测试用例的比例
- α：平衡参数（默认0.5）

## 文件结构

```
experiments/
├── step2_btp_experiment_openai.py     # 主要的BTP实验类
├── run_btp_openai_experiment.py       # 运行脚本
└── results/
    └── btp_openai/                     # 实验结果目录

test_openai_btp.py                      # 测试脚本
README_OpenAI_BTP.md                   # 本文档
```

## 准备工作

### 1. 安装依赖

确保已安装必要的Python包：

```bash
pip install requests numpy tqdm
```

### 2. 设置OpenAI API密钥

获取OpenAI API密钥：
1. 访问 https://platform.openai.com/api-keys
2. 创建或获取API密钥

设置环境变量：

```bash
# Linux/Mac
export OPENAI_API_KEY='your-api-key-here'

# Windows
set OPENAI_API_KEY=your-api-key-here
```

或者在代码中直接设置（不推荐用于生产环境）：

```python
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

## 使用方法

### 1. 快速测试

首先运行测试脚本确保一切配置正确：

```bash
python test_openai_btp.py
```

此脚本会：
- 测试OpenAI API连接
- 测试MBPP数据加载
- 运行简单的BTP测试

### 2. 运行完整实验

#### 方法一：使用运行脚本

```bash
python experiments/run_btp_openai_experiment.py
```

#### 方法二：直接运行BTP实验

```bash
python experiments/step2_btp_experiment_openai.py --problems 10 --candidates 3
```

### 3. 命令行参数

```bash
python experiments/step2_btp_experiment_openai.py \
    --model gpt-3.5-turbo \
    --dataset mbpp \
    --problems 50 \
    --candidates 5 \
    --output-dir experiments/results/my_btp_experiment \
    --api-key your-api-key \
    --base-url https://api.openai.com/v1
```

参数说明：
- `--model`: OpenAI模型名称（如gpt-3.5-turbo, gpt-4）
- `--dataset`: 数据集名称（目前支持mbpp）
- `--problems`: 处理的问题数量
- `--candidates`: 每个问题生成的候选解决方案数量
- `--output-dir`: 结果输出目录
- `--api-key`: OpenAI API密钥（可选，优先使用环境变量）
- `--base-url`: OpenAI API基础URL（可选）

## 实验流程

### 1. 基础生成阶段

对每个编程问题：
1. 格式化问题为提示词
2. 使用OpenAI API生成多个候选解决方案
3. 对每个候选方案进行测试
4. 将经验添加到回放缓冲区

### 2. 经验增强阶段

如果基础生成未完全成功：
1. 从缓冲区采样高P2Value的经验
2. 构造增强提示词（包含相关经验）
3. 生成新的候选解决方案
4. 测试并添加到缓冲区

### 3. 结果分析

实验完成后会生成：
- 总体成功率统计
- 经验缓冲区分析
- P2Value分布统计
- 详细的JSON结果文件

## 结果解读

### 成功率指标

- **总体成功率**: 完全通过所有测试的问题比例
- **基础成功率**: 仅使用基础生成的成功率
- **增强成功率**: 使用经验增强后的额外成功率

### P2Value分析

- **平均P2Value**: 所有经验的平均P2Value分数
- **通过率分布**: 测试通过率的统计分布
- **可能性分布**: 生成可能性分数的统计分布

### 经验回放效果

- **总经验数**: 收集的经验总数
- **增强解决方案比例**: 通过经验回放生成的解决方案比例
- **缓冲区利用率**: 经验回放的有效性

## 优化建议

### 1. 模型选择

- **gpt-3.5-turbo**: 成本较低，适合初步实验
- **gpt-4**: 性能更好，但成本较高

### 2. 超参数调优

- **候选数量**: 更多候选通常带来更好的结果，但增加成本
- **温度设置**: 基础生成使用0.8，增强生成使用0.7
- **问题数量**: 根据预算和实验需求调整

### 3. 成本控制

- 从少量问题开始测试（如10个问题）
- 监控API使用量和成本
- 适当设置API调用间隔避免限流

## 故障排除

### 常见问题

1. **API密钥错误**
   - 检查环境变量设置
   - 确认API密钥有效性
   - 检查账户余额

2. **网络连接问题**
   - 确认网络连接正常
   - 检查防火墙设置
   - 考虑使用代理

3. **导入错误**
   - 确认项目路径设置正确
   - 检查所有依赖是否安装
   - 验证Python版本兼容性

4. **测试失败**
   - 检查MBPP数据是否正确加载
   - 确认测试环境配置
   - 查看详细错误日志

### 调试方法

启用调试模式：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

查看详细输出：

```bash
python test_openai_btp.py 2>&1 | tee debug.log
```

## 进阶使用

### 自定义提示词

修改 `format_prompt` 和 `format_prioritized_prompt` 方法来自定义提示词格式。

### 扩展数据集

可以扩展支持其他编程数据集，需要实现相应的数据加载和测试函数。

### 集成其他模型

可以修改 `OpenAIClient` 类来支持其他兼容OpenAI API的模型服务。

## 示例输出

```
使用模型: gpt-3.5-turbo
处理问题数: 10
候选数量: 3

============================================================
BTP OpenAI实验开始
============================================================

开始收集经验数据... (问题数量: 10, 候选数量: 3)

处理问题 11: Remove extra spaces...
  基础生成未完全成功，尝试经验增强生成...
  经验增强生成成功！

...

数据收集完成！成功率: 70.00% (7/10)

缓冲区分析:
  总经验数: 45
  完全成功率: 31.11%
  增强解决方案比例: 20.00%
  平均P2Value: 0.456
  平均通过率: 0.623

============================================================
实验完成！
============================================================
总体成功率: 70.00%
处理的问题数: 10
每个问题的候选数: 3
```

## 结论

OpenAI BTP实验提供了一个强大的框架来评估和改进基于大语言模型的代码生成能力。通过结合束搜索、测试反馈和经验回放，可以显著提高代码生成的成功率和质量。 