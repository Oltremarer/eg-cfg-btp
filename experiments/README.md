# BTP 实验套件

基于论文《Enhancing LLMs for Code Generation with Possibility and Pass rate Prioritized Experience Replay》的完整实验实现。

## 实验概述

本实验套件包含四个核心实验：

1. **基线实验** (`step1_baseline_experiment.py`) - 评测标准采样方法的性能
2. **BTP实验** (`step2_btp_experiment.py`) - 验证Beam Search + Testing + Prioritized Experience Replay的有效性
3. **消融研究** (`step3_ablation_study.py`) - 对比不同回放策略的效果
4. **超参数分析** (`step4_hyperparameter_study.py`) - 探索关键超参数的影响

## 快速开始

### 运行完整实验套件

```bash
# 使用默认参数运行所有实验
python experiments/run_all_experiments.py --model_name "deepseek-ai/deepseek-coder-1.3b-instruct"

# 快速测试模式（小规模实验）
python experiments/run_all_experiments.py --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" --mode quick

# 指定数据集
python experiments/run_all_experiments.py --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" --dataset humaneval
```

### 运行单个实验

```bash
# 只运行基线实验
python experiments/run_all_experiments.py --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" --mode single --single_step baseline

# 只运行BTP实验
python experiments/run_all_experiments.py --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" --mode single --single_step btp
```

## 详细使用说明

### 1. 基线实验

评测标准的采样-过滤方法在代码生成任务上的性能。

```bash
python experiments/step1_baseline_experiment.py \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset mbpp \
    --num_samples 10 \
    --max_problems 50 \
    --output_dir experiments/results/baseline
```

**参数说明：**
- `--num_samples`: 每个问题生成的解决方案数量
- `--max_problems`: 评测的问题总数
- `--temperature`: 采样温度

**输出：**
- Pass@k 指标
- 详细的生成结果和测试结果

### 2. BTP实验

实现完整的BTP管道：Beam Search采样 → 测试评估 → P2Value计算 → 优先经验回放。

```bash
python experiments/step2_btp_experiment.py \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset mbpp \
    --collect_problems 100 \
    --output_dir experiments/results/btp
```

**核心组件：**
- **P2ValueCalculator**: 计算综合生成概率和通过率的指标
- **ExperienceReplayBuffer**: 存储和优先采样失败经验
- **Beam Search**: 生成多样化候选解决方案并计算概率

**输出：**
- 经验回放缓冲区分析
- P2Value分布统计
- 采样策略效果对比

### 3. 消融研究

对比不同经验回放策略的效果，证明P2Value优先采样的优势。

```bash
python experiments/step3_ablation_study.py \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset mbpp \
    --num_problems 30 \
    --output_dir experiments/results/ablation
```

**对比策略：**
- 基线方法（无回放）
- P2Value优先回放
- 随机回放

**输出：**
- 各策略的性能对比
- 统计显著性分析
- 改进幅度量化

### 4. 超参数分析

探索关键超参数对BTP性能的影响，找到最优配置。

```bash
python experiments/step4_hyperparameter_study.py \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset mbpp \
    --search_type focused \
    --max_configs 15 \
    --output_dir experiments/results/hyperparameter
```

**关键超参数：**
- `alpha`: P2Value中平衡概率和通过率的权重
- `buffer_size`: 经验回放缓冲区大小
- `num_beams`: Beam Search宽度
- `temperature`: 采样温度
- `batch_size`: 回放批次大小

**搜索策略：**
- `grid`: 网格搜索所有参数组合
- `focused`: 针对关键参数的重点搜索

## 实验配置

### 支持的模型

```python
# 本地模型
"deepseek-ai/deepseek-coder-1.3b-instruct"
"deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

# 推理端点（需要配置）
"inference_endpoint_model_name"
```

### 支持的数据集

- **MBPP**: Mostly Basic Python Problems
- **HumanEval**: 人工评估的Python编程问题
- **APPS**: 竞赛级编程问题（实验性支持）

### 实验参数调优建议

**快速测试：**
```bash
--num_samples 3 --max_problems 10 --collect_problems 10
```

**标准实验：**
```bash
--num_samples 10 --max_problems 50 --collect_problems 50
```

**完整评估：**
```bash
--num_samples 20 --max_problems 100 --collect_problems 100
```

## 结果分析

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

### 结果解读

- **基线 vs BTP**: BTP应该在Pass@k和Success Rate上显著优于基线
- **BTP vs 随机回放**: P2Value优先采样应该优于随机采样
- **超参数影响**: alpha参数通常在0.5左右效果最好

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减少问题数量和批次大小
   --max_problems 20 --collect_problems 20 --batch_size 10
   ```

2. **模型加载失败**
   ```bash
   # 检查模型名称和访问权限
   # 确保有足够的GPU内存
   ```

3. **数据集加载错误**
   ```bash
   # 确保网络连接正常
   # 检查数据集名称拼写
   ```

### 调试模式

```bash
# 启用详细输出
export PYTHONPATH="${PYTHONPATH}:."
python -u experiments/step1_baseline_experiment.py --model_name "your_model" --dataset mbpp --max_problems 5
```

## 扩展实验

### 添加新数据集

1. 在相应的utils文件中添加数据加载函数
2. 修改实验脚本中的数据集选择逻辑
3. 适配测试用例格式

### 添加新评估指标

1. 在实验脚本中添加指标计算函数
2. 修改结果分析和报告生成逻辑
3. 更新输出格式

### 自定义P2Value计算

```python
class CustomP2ValueCalculator(P2ValueCalculator):
    def calculate_p2value(self, log_prob, sequence_length, passed_tests, total_tests):
        # 自定义计算逻辑
        pass
```

## 引用

如果使用本实验代码，请引用原论文：

```bibtex
@article{your_paper_2026,
  title={Enhancing LLMs for Code Generation with Possibility and Pass rate Prioritized Experience Replay},
  author={Your Authors},
  journal={AAAI},
  year={2026}
}
``` 