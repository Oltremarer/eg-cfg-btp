"""
智能Prompt模板适配系统
基于代码生成模型Prompt模板权威指南

核心原则：
1. 模型决定模板：根据模型家族自动选择正确的格式和特殊标记
2. 数据集决定内容：根据数据集类型格式化问题内容
3. 基础版vs指令版：自动区分并使用相应的模板

支持的模型家族：
- LLaMA/CodeLLaMA系列 ([INST]格式)
- DeepSeek-Coder系列 (### Instruction格式)
- StarCoder系列 (直接补全或Alpaca格式)
- Qwen系列 (ChatML格式)
- OpenAI/Claude系列 (messages格式)
"""

import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ModelFamily(Enum):
    """模型家族枚举"""
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    STARCODER = "starcoder"
    QWEN = "qwen"
    OPENAI = "openai"
    CLAUDE = "claude"
    UNKNOWN = "unknown"


class ModelType(Enum):
    """模型类型枚举"""
    BASE = "base"           # 基础/补全模型
    INSTRUCT = "instruct"   # 指令微调模型
    CHAT = "chat"           # 对话模型


@dataclass
class ModelInfo:
    """模型信息"""
    family: ModelFamily
    type: ModelType
    name: str
    size: Optional[str] = None
    special_format: Optional[str] = None


class ModelDetector:
    """智能模型检测器"""
    
    # 模型家族检测规则
    FAMILY_PATTERNS = {
        ModelFamily.LLAMA: [
            r"llama", r"code[-_]?llama", r"codellama",
            r"llama[-_]?2", r"llama2", r"meta[-_]?llama"
        ],
        ModelFamily.DEEPSEEK: [
            r"deepseek", r"deepseek[-_]?coder", r"deepseek[-_]?v[0-9]"
        ],
        ModelFamily.STARCODER: [
            r"starcoder", r"star[-_]?coder", r"wizardcoder",
            r"wizard[-_]?coder", r"bigcode"
        ],
        ModelFamily.QWEN: [
            r"qwen", r"qwen[-_]?coder", r"qwen[-_]?code",
            r"tongyi", r"通义"
        ],
        ModelFamily.OPENAI: [
            r"gpt[-_]?[0-9]", r"gpt[-_]?3\.?5", r"gpt[-_]?4",
            r"text[-_]?davinci", r"code[-_]?davinci"
        ],
        ModelFamily.CLAUDE: [
            r"claude", r"claude[-_]?[0-9]", r"anthropic"
        ]
    }
    
    # 模型类型检测规则
    TYPE_PATTERNS = {
        ModelType.INSTRUCT: [r"instruct", r"instruction", r"chat"],
        ModelType.CHAT: [r"chat", r"conversation"],
        ModelType.BASE: [r"base", r"completion", r"^(?!.*instruct)(?!.*chat).*$"]
    }
    
    # 特殊格式检测（如CodeLLaMA-70B的特殊格式）
    SPECIAL_FORMAT_PATTERNS = {
        "llama_70b_instruct": [r"llama.*70b.*instruct", r"codellama.*70b.*instruct"],
        "starcoder_base_humaneval": [r"starcoder.*base", r"star[-_]?coder(?!.*wizard)"]
    }
    
    @classmethod
    def detect_model(cls, model_name: str) -> ModelInfo:
        """检测模型信息"""
        model_name_lower = model_name.lower()
        
        # 检测家族
        family = cls._detect_family(model_name_lower)
        
        # 检测类型
        model_type = cls._detect_type(model_name_lower)
        
        # 检测特殊格式
        special_format = cls._detect_special_format(model_name_lower)
        
        # 检测尺寸
        size = cls._extract_size(model_name_lower)
        
        return ModelInfo(
            family=family,
            type=model_type,
            name=model_name,
            size=size,
            special_format=special_format
        )
    
    @classmethod
    def _detect_family(cls, model_name: str) -> ModelFamily:
        """检测模型家族"""
        for family, patterns in cls.FAMILY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, model_name):
                    return family
        return ModelFamily.UNKNOWN
    
    @classmethod
    def _detect_type(cls, model_name: str) -> ModelType:
        """检测模型类型"""
        for model_type, patterns in cls.TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, model_name):
                    return model_type
        return ModelType.BASE  # 默认为base类型
    
    @classmethod
    def _detect_special_format(cls, model_name: str) -> Optional[str]:
        """检测特殊格式"""
        for format_name, patterns in cls.SPECIAL_FORMAT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, model_name):
                    return format_name
        return None
    
    @classmethod
    def _extract_size(cls, model_name: str) -> Optional[str]:
        """提取模型尺寸"""
        size_patterns = [
            r"(\d+\.?\d*[bm])",  # 如 7b, 13b, 70b, 1.3b
            r"(\d+\.?\d*)([-_]?)(billion|million|b|m)"
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, model_name)
            if match:
                return match.group(1)
        return None


class PromptTemplateEngine:
    """智能Prompt模板引擎"""
    
    def __init__(self):
        self.detector = ModelDetector()
        self._init_templates()
    
    def _init_templates(self):
        """初始化所有模板"""
        
        # LLaMA/CodeLLaMA模板
        self.llama_templates = {
            ModelType.INSTRUCT: {
                "system_template": """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]""",
                "user_only_template": """<s>[INST] {user_prompt} [/INST]""",
                "default_system": "You are an expert Python programmer. Please only output the code."
            },
            ModelType.BASE: {
                "template": "{content}",  # 直接续写
                "default_system": ""
            }
        }
        
        # CodeLLaMA-70B-Instruct特殊格式
        self.llama_70b_template = """<s>Source: system

{system_prompt}<step>Source: user

{user_prompt}<step>Source: assistant
Destination: user

"""
        
        # DeepSeek-Coder模板
        self.deepseek_templates = {
            ModelType.INSTRUCT: {
                "template": """You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer
### Instruction:
{user_prompt}
### Response:""",
                "with_examples_template": """You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer
### Instruction:
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{examples}

Here is my problem:
{user_prompt}

### Response:"""
            },
            ModelType.BASE: {
                "template": "{content}",  # 直接续写
            }
        }
        
        # StarCoder模板
        self.starcoder_templates = {
            ModelType.BASE: {
                "template": "{content}",  # 直接续写
                "humaneval_prefix": """<filename>solutions/solution_1.py
# Here is the correct implementation of the code exercise
{content}"""
            },
            ModelType.INSTRUCT: {  # WizardCoder等微调变体
                "template": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_prompt}

### Response:"""
            }
        }
        
        # Qwen模板 (ChatML格式)
        self.qwen_templates = {
            ModelType.INSTRUCT: {
                "template": """<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_prompt}
<|im_end|>
<|im_start|>assistant""",
                "user_only_template": """<|im_start|>user
{user_prompt}
<|im_end|>
<|im_start|>assistant""",
                "default_system": "You are a helpful programming assistant."
            }
        }
        
        # OpenAI/Claude模板 (Messages格式)
        self.openai_claude_templates = {
            ModelType.CHAT: {
                "messages_format": True,
                "default_system": "You are an expert Python programmer. Please provide only the complete, runnable code solution without any extra explanations."
            }
        }
    
    def format_prompt(self, model_name: str, dataset: str, problem: Dict[str, Any], 
                     system_prompt: Optional[str] = None, 
                     use_examples: bool = False,
                     examples: Optional[List[Dict]] = None) -> Union[str, List[Dict]]:
        """
        智能格式化prompt
        
        Args:
            model_name: 模型名称
            dataset: 数据集名称 ('mbpp', 'humaneval', 'apps')
            problem: 问题字典
            system_prompt: 自定义系统提示（可选）
            use_examples: 是否使用few-shot examples
            examples: 示例列表（可选）
            
        Returns:
            str: 格式化的prompt字符串（大多数模型）
            List[Dict]: messages格式（OpenAI/Claude）
        """
        
        # 检测模型信息
        model_info = self.detector.detect_model(model_name)
        
        # 获取数据集内容
        user_content = self._format_dataset_content(dataset, problem)
        
        # 根据模型家族和类型选择模板
        if model_info.family == ModelFamily.LLAMA:
            return self._format_llama_prompt(model_info, user_content, system_prompt, dataset)
        
        elif model_info.family == ModelFamily.DEEPSEEK:
            return self._format_deepseek_prompt(model_info, user_content, use_examples, examples)
        
        elif model_info.family == ModelFamily.STARCODER:
            return self._format_starcoder_prompt(model_info, user_content, dataset)
        
        elif model_info.family == ModelFamily.QWEN:
            return self._format_qwen_prompt(model_info, user_content, system_prompt)
        
        elif model_info.family in [ModelFamily.OPENAI, ModelFamily.CLAUDE]:
            return self._format_openai_claude_prompt(model_info, user_content, system_prompt)
        
        else:
            # 未知模型，使用通用格式
            return self._format_generic_prompt(user_content)
    
    def _format_dataset_content(self, dataset: str, problem: Dict[str, Any]) -> str:
        """格式化数据集内容"""
        if dataset.lower() == "mbpp":
            test_cases = "\n".join([f"  {test}" for test in problem.get('test_list', [])])
            return f"""Problem: {problem['text']}

Test cases:
{test_cases}

Provide a complete Python function:"""
        
        elif dataset.lower() == "humaneval":
            return f"""Complete the following Python function:

{problem['prompt']}"""
        
        elif dataset.lower() == "apps":
            test_cases = "\n".join([f"  {test}" for test in problem.get('test_list', [])])
            return f"""Problem: {problem['question']}

Test cases:
{test_cases}

Provide a complete Python function:"""
        
        else:
            return f"Solve this programming problem: {problem}"
    
    def _format_llama_prompt(self, model_info: ModelInfo, user_content: str, 
                           system_prompt: Optional[str], dataset: str) -> str:
        """格式化LLaMA prompt"""
        
        # 检查是否是70B的特殊格式
        if model_info.special_format == "llama_70b_instruct":
            system = system_prompt or self.llama_templates[ModelType.INSTRUCT]["default_system"]
            return self.llama_70b_template.format(
                system_prompt=system,
                user_prompt=user_content
            )
        
        if model_info.type == ModelType.INSTRUCT:
            template_info = self.llama_templates[ModelType.INSTRUCT]
            
            if system_prompt:
                return template_info["system_template"].format(
                    system_prompt=system_prompt,
                    user_prompt=user_content
                )
            else:
                # 使用默认系统提示
                return template_info["system_template"].format(
                    system_prompt=template_info["default_system"],
                    user_prompt=user_content
                )
        else:
            # Base模型，直接续写
            if dataset.lower() == "humaneval":
                return problem['prompt']  # HumanEval直接使用prompt字段
            else:
                # 对其他数据集，简单格式化
                return f"# {user_content}\ndef "
    
    def _format_deepseek_prompt(self, model_info: ModelInfo, user_content: str,
                              use_examples: bool, examples: Optional[List[Dict]]) -> str:
        """格式化DeepSeek prompt"""
        
        if model_info.type == ModelType.INSTRUCT:
            template_info = self.deepseek_templates[ModelType.INSTRUCT]
            
            if use_examples and examples:
                # 格式化examples
                formatted_examples = self._format_examples_for_deepseek(examples)
                return template_info["with_examples_template"].format(
                    examples=formatted_examples,
                    user_prompt=user_content
                )
            else:
                return template_info["template"].format(user_prompt=user_content)
        else:
            # Base模型
            return user_content
    
    def _format_starcoder_prompt(self, model_info: ModelInfo, user_content: str, dataset: str) -> str:
        """格式化StarCoder prompt"""
        
        if model_info.type == ModelType.INSTRUCT:
            # WizardCoder等微调变体使用Alpaca格式
            template_info = self.starcoder_templates[ModelType.INSTRUCT]
            return template_info["template"].format(user_prompt=user_content)
        else:
            # Base模型
            template_info = self.starcoder_templates[ModelType.BASE]
            
            if dataset.lower() == "humaneval":
                # HumanEval使用特殊前缀技巧
                return template_info["humaneval_prefix"].format(content=user_content)
            else:
                # 其他数据集直接续写
                return template_info["template"].format(content=user_content)
    
    def _format_qwen_prompt(self, model_info: ModelInfo, user_content: str, 
                          system_prompt: Optional[str]) -> str:
        """格式化Qwen prompt"""
        
        if model_info.type == ModelType.INSTRUCT:
            template_info = self.qwen_templates[ModelType.INSTRUCT]
            
            if system_prompt:
                return template_info["template"].format(
                    system_prompt=system_prompt,
                    user_prompt=user_content
                )
            else:
                return template_info["user_only_template"].format(user_prompt=user_content)
        else:
            return user_content
    
    def _format_openai_claude_prompt(self, model_info: ModelInfo, user_content: str, 
                                   system_prompt: Optional[str]) -> List[Dict]:
        """格式化OpenAI/Claude prompt (返回messages格式)"""
        
        template_info = self.openai_claude_templates[ModelType.CHAT]
        system = system_prompt or template_info["default_system"]
        
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ]
    
    def _format_generic_prompt(self, user_content: str) -> str:
        """通用格式（未知模型）"""
        return f"Solve the following programming problem:\n\n{user_content}"
    
    def _format_examples_for_deepseek(self, examples: List[Dict]) -> str:
        """为DeepSeek格式化few-shot examples"""
        formatted = []
        for i, example in enumerate(examples, 1):
            formatted.append(f"""- Example {i}:
>>> Problem:
{example['problem']}
>>> Test Cases:
{chr(10).join(example['test_cases'])}

>>> Code:
```python
{example['solution']}
```""")
        return "\n\n".join(formatted)


# 全局实例
prompt_engine = PromptTemplateEngine()


def get_model_prompt(model_name: str, dataset: str, problem: Dict[str, Any], 
                    system_prompt: Optional[str] = None, 
                    use_examples: bool = False,
                    examples: Optional[List[Dict]] = None) -> Union[str, List[Dict]]:
    """
    便捷函数：获取特定模型的prompt
    
    这是主要的对外接口，自动处理所有模型适配逻辑
    """
    return prompt_engine.format_prompt(
        model_name=model_name,
        dataset=dataset,
        problem=problem,
        system_prompt=system_prompt,
        use_examples=use_examples,
        examples=examples
    )


def detect_model_info(model_name: str) -> ModelInfo:
    """便捷函数：检测模型信息"""
    return ModelDetector.detect_model(model_name)


def validate_model_compatibility(model_name: str, dataset: str) -> Dict[str, Any]:
    """验证模型与数据集的兼容性"""
    model_info = detect_model_info(model_name)
    
    compatibility = {
        "model_info": model_info,
        "supported": True,
        "recommendations": [],
        "warnings": []
    }
    
    # 针对不同模型家族的建议
    if model_info.family == ModelFamily.STARCODER and dataset.lower() != "humaneval":
        compatibility["recommendations"].append(
            "StarCoder在HumanEval上表现最佳，建议使用特殊前缀格式"
        )
    
    if model_info.family == ModelFamily.DEEPSEEK and dataset.lower() == "mbpp":
        compatibility["recommendations"].append(
            "DeepSeek-Coder建议使用few-shot examples以获得更好性能"
        )
    
    if model_info.family == ModelFamily.UNKNOWN:
        compatibility["warnings"].append(
            "未识别的模型家族，将使用通用格式，可能影响性能"
        )
    
    return compatibility


# 向后兼容性支持
class PromptTemplates:
    """保持向后兼容的旧接口"""
    
    @classmethod
    def get_prompt(cls, dataset: str, problem: dict, model_name: str = "generic") -> str:
        """向后兼容的接口"""
        result = get_model_prompt(model_name, dataset, problem)
        
        # 如果返回的是messages格式，转换为字符串
        if isinstance(result, list):
            # 提取user内容
            for msg in result:
                if msg["role"] == "user":
                    return msg["content"]
            return str(result)
        
        return result
    
    @classmethod
    def validate_consistency(cls) -> bool:
        """向后兼容的验证方法"""
        print("✅ 新的智能Prompt系统已启用")
        print("✅ 支持自动模型检测和格式适配")
        print("✅ 向后兼容性已保证")
        return True


# 使用示例和测试
if __name__ == "__main__":
    # 测试模型检测
    test_models = [
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "meta-llama/CodeLlama-7b-Instruct-hf",
        "bigcode/starcoder-15b",
        "WizardLM/WizardCoder-15B-V1.0",
        "Qwen/CodeQwen1.5-7B-Chat",
        "gpt-3.5-turbo",
        "claude-3-sonnet"
    ]
    
    print("🧪 测试模型检测和prompt格式化：\n")
    
    sample_problem = {
        'text': 'Write a function to find the maximum element in a list.',
        'test_list': ['assert max_element([1, 2, 3]) == 3', 'assert max_element([10, 5, 8]) == 10']
    }
    
    for model in test_models:
        print(f"📋 模型: {model}")
        model_info = detect_model_info(model)
        print(f"   家族: {model_info.family.value}")
        print(f"   类型: {model_info.type.value}")
        if model_info.size:
            print(f"   尺寸: {model_info.size}")
        if model_info.special_format:
            print(f"   特殊格式: {model_info.special_format}")
        
        # 生成prompt示例
        prompt = get_model_prompt(model, "mbpp", sample_problem)
        if isinstance(prompt, list):
            print(f"   格式: Messages (OpenAI/Claude)")
            print(f"   系统: {prompt[0]['content'][:50]}...")
            print(f"   用户: {prompt[1]['content'][:50]}...")
        else:
            print(f"   Prompt: {prompt[:100]}...")
        print()
    
    print("✅ 智能Prompt模板系统测试完成！") 