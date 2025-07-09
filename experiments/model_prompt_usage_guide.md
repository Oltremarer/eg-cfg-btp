# 智能Prompt模板系统使用指南

## 概述

基于《代码生成模型Prompt模板权威指南》，我们为EG-CFG项目开发了智能Prompt适配系统。该系统能够：

- **自动检测模型家族**并选择正确的prompt格式
- **优化生成参数**以获得最佳性能
- **统一管理**所有模型配置和模板
- **保持向后兼容**并方便扩展

## 核心原则

### 1. 模型决定模板
不同模型家族有不同的格式要求：

| 模型家族 | 格式示例 | 特点 |
|---------|---------|------|
| **LLaMA/CodeLLaMA** | `[INST]`格式 | 70B有特殊Source格式 |
| **DeepSeek-Coder** | `### Instruction/Response` | 固定格式，必须遵守 |
| **StarCoder** | 直接续写 vs Alpaca | base版在HumanEval上有特殊技巧 |
| **Qwen** | `<\|im_start\|>` ChatML | 建议用官方tokenizer |
| **OpenAI/Claude** | `messages`列表 | 必须用system消息约束输出 |

### 2. 数据集决定内容
- **MBPP**: 问题描述 + 测试用例格式
- **HumanEval**: 函数签名 + 文档字符串
- **APPS**: 复杂问题描述 + 多测试用例

### 3. 基础版vs指令版
自动识别并使用相应的模板格式。

## 快速开始

### 基本使用

```python
from experiments.prompt_templates import get_model_prompt, detect_model_info
from experiments.shared.model_configs import get_model_config, get_optimal_generation_params

# 1. 检测模型信息
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
model_info = detect_model_info(model_name)
print(f"家族: {model_info.family.value}, 类型: {model_info.type.value}")

# 2. 获取优化配置
config = get_model_config(model_name)
params = get_optimal_generation_params(model_name, "mbpp")

# 3. 生成适配的prompt
problem = {
    'text': 'Write a function to find the maximum element in a list.',
    'test_list': ['assert max_element([1, 2, 3]) == 3']
}

prompt = get_model_prompt(
    model_name=model_name,
    dataset="mbpp",
    problem=problem
)
print(prompt)
```

### 高级使用（Few-shot Examples）

```python
# 为DeepSeek等模型使用few-shot examples
examples = [
    {
        "problem": "Write a function to find similar elements from two tuple lists.",
        "test_cases": ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)"],
        "solution": "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return res"
    }
]

prompt = get_model_prompt(
    model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
    dataset="mbpp",
    problem=problem,
    use_examples=True,
    examples=examples
)
```

## 不同模型家族的具体格式

### 1. DeepSeek-Coder系列

**模型检测**: `deepseek`, `deepseek-coder`, `deepseek-v*`

**Instruct格式**:
```
You are an AI programming assistant, utilizing the Deepseek Coder model...
### Instruction:
{user_prompt}
### Response:
```

**使用建议**:
- MBPP数据集建议使用few-shot examples
- 温度设置0.6-0.7获得最佳效果
- 支持高达8K上下文长度

### 2. LLaMA/CodeLLaMA系列

**模型检测**: `llama`, `codellama`, `meta-llama`

**Instruct格式**:
```
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]
```

**70B特殊格式**:
```
<s>Source: system

{system_prompt}<step>Source: user

{user_prompt}<step>Source: assistant
Destination: user

```

**使用建议**:
- 建议添加系统提示约束输出
- Base版本在HumanEval上直接使用prompt字段
- 13B及以上模型表现更稳定

### 3. StarCoder系列

**模型检测**: `starcoder`, `wizardcoder`, `bigcode`

**Base格式（HumanEval优化）**:
```
<filename>solutions/solution_1.py
# Here is the correct implementation of the code exercise
{content}
```

**WizardCoder（Alpaca格式）**:
```
Below is an instruction that describes a task...

### Instruction:
{user_prompt}

### Response:
```

**使用建议**:
- Base版本在HumanEval上使用特殊前缀
- WizardCoder使用Alpaca格式而非原始StarCoder格式
- 温度设置0.5-0.7

### 4. Qwen系列

**模型检测**: `qwen`, `qwen-coder`, `tongyi`

**ChatML格式**:
```
<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_prompt}
<|im_end|>
<|im_start|>assistant
```

**使用建议**:
- 强烈建议使用官方tokenizer的apply_chat_template
- 支持长上下文（8K+）
- 中英文混合数据集表现优秀

### 5. OpenAI/Claude系列

**格式（Messages）**:
```python
[
    {"role": "system", "content": "You are an expert Python programmer..."},
    {"role": "user", "content": "{user_prompt}"}
]
```

**使用建议**:
- **必须**使用system消息约束输出格式
- 避免模型生成解释性文字影响评测
- GPT-4建议降低温度到0.5-0.6

## 实验文件集成

### 更新现有实验

```python
# 原有方式
def format_prompt(self, problem):
    return f"Solve: {problem['text']}"

# 新的智能方式
from experiments.prompt_templates import get_model_prompt
from experiments.shared.model_configs import get_model_config

def format_prompt(self, problem):
    # 自动适配模型格式
    return get_model_prompt(
        model_name=self.model_name,
        dataset="mbpp",
        problem=problem,
        use_examples=self.model_config.use_examples
    )
```

### 参数优化

```python
from experiments.shared.model_configs import get_optimal_generation_params

# 获取数据集特定的最优参数
params = get_optimal_generation_params(model_name, "mbpp")
# 输出: {'temperature': 0.7, 'max_tokens': 512, 'use_examples': True, ...}

# 在生成时使用
outputs = model.generate(prompt, **params)
```

## 性能建议

### 模型推荐

**MBPP数据集**（准确率优先）:
1. `deepseek-ai/deepseek-coder-6.7b-instruct` - 高准确率
2. `gpt-4` - 最高准确率（成本高）
3. `meta-llama/CodeLlama-7b-Instruct-hf` - 平衡性能

**HumanEval数据集**（准确率优先）:
1. `gpt-4` - 业界标杆
2. `claude-3-sonnet` - 强大推理能力
3. `bigcode/starcoder-15b` - HumanEval优化

**APPS数据集**（复杂问题）:
1. `gpt-4` - 复杂问题处理能力强
2. `claude-3-sonnet` - 优秀推理能力
3. `meta-llama/CodeLlama-13b-Instruct-hf` - 大模型优势

### 参数调优建议

| 数据集 | 温度范围 | Max Tokens | Few-shot | 特殊考虑 |
|--------|----------|------------|----------|----------|
| **MBPP** | 0.6-0.8 | 256-512 | 推荐（DeepSeek） | 简单问题，避免过度生成 |
| **HumanEval** | 0.5-0.7 | 256-512 | 不推荐 | 精确补全，低温度 |
| **APPS** | 0.7-0.9 | 1024-2048 | 可选 | 复杂问题，需要更多token |

## 故障排除

### 常见问题

1. **Unknown模型家族**
   ```python
   # 检查模型信息
   model_info = detect_model_info("your-model-name")
   if model_info.family == ModelFamily.UNKNOWN:
       print("需要添加模型检测规则")
   ```

2. **API密钥问题**
   ```python
   validation = validate_model_setup("gpt-4")
   if validation["warnings"]:
       print("配置问题:", validation["warnings"])
   ```

3. **性能不佳**
   ```python
   # 检查兼容性建议
   compatibility = validate_model_compatibility("model-name", "mbpp")
   print("优化建议:", compatibility["recommendations"])
   ```

### 调试技巧

```python
# 1. 检查生成的prompt
prompt = get_model_prompt(model_name, dataset, problem)
print("Generated prompt:", prompt[:200])

# 2. 验证参数设置
params = get_optimal_generation_params(model_name, dataset)
print("Optimal params:", params)

# 3. 查看模型信息
model_info = detect_model_info(model_name)
print(f"Family: {model_info.family}, Type: {model_info.type}")
```

## 扩展指南

### 添加新模型家族

1. 在`experiments/prompt_templates.py`中添加检测规则：
   ```python
   FAMILY_PATTERNS = {
       ModelFamily.NEW_FAMILY: [r"pattern1", r"pattern2"]
   }
   ```

2. 在`experiments/shared/model_configs.py`中添加配置：
   ```python
   self.register_config(ModelConfig(
       name="new-model",
       family=ModelFamily.NEW_FAMILY,
       # ... 其他配置
   ))
   ```

### 添加新数据集

1. 在`_format_dataset_content`方法中添加格式化逻辑
2. 在模型配置中添加数据集特定参数
3. 更新推荐算法

## 版本更新

### v1.0 特性
- ✅ 自动模型检测
- ✅ 智能prompt适配
- ✅ 参数优化
- ✅ 向后兼容

### 计划特性
- 🔄 更多模型家族支持
- 🔄 动态参数调优
- 🔄 A/B测试框架
- 🔄 性能监控

---

**注意**: 该系统设计为渐进式改进。现有代码可以逐步迁移，无需一次性重写所有实验文件。 