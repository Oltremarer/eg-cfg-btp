# 基于特定模型的EG-CFG实验计划

## 🎯 实验目标
基于EG-CFG框架，使用大模型（source）采样来提升小模型（target）的代码生成能力。

## 🏗️ 模型配置方案

### 1. 大模型（Source Models）- 用于采样
```
• StarCoder2-15B          # 开源代码生成SOTA
• Qwen2.5-Coder-32B       # 阿里通义千问代码模型  
• CodeLlama-34B           # Meta的代码专用模型
• DeepSeek-V3-0324        # DeepSeek最新大模型
```

### 2. 小模型（Target Models）- 用于微调
```
• StarCoder2-3B           # StarCoder轻量版
• Qwen2.5-Coder-7B        # Qwen轻量版
• CodeLlama-7B            # CodeLlama轻量版  
• DeepSeek-Coder-1.3B     # DeepSeek轻量版
```

### 3. 实验配对方案
```
实验1: StarCoder2-15B → StarCoder2-3B
实验2: Qwen2.5-Coder-32B → Qwen2.5-Coder-7B  
实验3: CodeLlama-34B → CodeLlama-7B
实验4: DeepSeek-V3-0324 → DeepSeek-Coder-1.3B
实验5: 跨家族实验（如 StarCoder2-15B → DeepSeek-Coder-1.3B）
```

## 🧪 第一个实验：同族模型增强

### 实验设计
- **数据集**: MBPP (500题) + HumanEval (164题)
- **方法**: Beam Search + Testing + Prioritized Experience Replay (BTP)
- **评估指标**: Pass@1, Pass@5, Pass@10

### 执行步骤

#### Step 1: 基线测试
```bash
# 测试大模型baseline
python experiments/step1_baseline_experiment.py \
  --model-name StarCoder2-15B \
  --dataset mbpp \
  --num-samples 10

# 测试小模型baseline  
python experiments/step1_baseline_experiment.py \
  --model-name StarCoder2-3B \
  --dataset mbpp \
  --num-samples 10
```

#### Step 2: BTP增强实验
```bash
python experiments/step2_btp_finetune_experiment.py \
  --source-model StarCoder2-15B \
  --target-model StarCoder2-3B \
  --mode finetune \
  --dataset mbpp \
  --max-problems 100 \
  --num-beams 5 \
  --n-iterations 3 \
  --sampling-method power \
  --sampling-alpha 1.5 \
  --use-lora \
  --output-dir ./results/starcoder_experiment
```

## 📊 预期输出
1. **性能对比报告**: 增强前后的Pass@k指标对比
2. **模型检查点**: 微调后的模型权重
3. **详细日志**: 每轮训练的P2Value分布和采样统计

## 🔄 下一步扩展
- 消融研究（Step3）: 测试BTP各组件的贡献
- 超参数优化（Step4）: 寻找最优配置
- 跨模型族实验: 验证方法的通用性 