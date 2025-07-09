"""
智能模型配置管理系统
配合prompt模板系统，为不同模型提供优化的参数设置
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from experiments.prompt_templates import ModelFamily, ModelType, detect_model_info


@dataclass
class ModelConfig:
    """模型配置信息"""
    name: str
    family: ModelFamily
    type: ModelType
    
    # 生成参数
    default_temperature: float = 0.8
    default_max_tokens: int = 512
    default_top_p: float = 0.95
    default_num_beams: int = 5
    
    # 数据集特定设置
    dataset_preferences: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 性能优化建议
    batch_size_recommendation: int = 1
    use_examples: bool = False
    preferred_examples_count: int = 3
    
    # API/推理配置
    api_endpoint: Optional[str] = None
    requires_api_key: bool = False
    supports_streaming: bool = False
    timeout_seconds: int = 30
    
    # 特殊处理标志
    needs_special_encoding: bool = False
    supports_system_prompt: bool = True
    max_context_length: int = 4096
    
    # 评估建议
    recommended_metrics: List[str] = field(default_factory=lambda: ["pass@1", "pass@5"])


class ModelConfigManager:
    """模型配置管理器"""
    
    def __init__(self):
        self.configs = {}
        self._init_predefined_configs()
    
    def _init_predefined_configs(self):
        """初始化预定义的模型配置"""
        
        # DeepSeek系列配置
        self.register_config(ModelConfig(
            name="deepseek-coder",
            family=ModelFamily.DEEPSEEK,
            type=ModelType.INSTRUCT,
            default_temperature=0.7,
            default_max_tokens=512,
            dataset_preferences={
                "mbpp": {
                    "use_examples": True,
                    "examples_count": 3,
                    "temperature": 0.6
                },
                "humaneval": {
                    "use_examples": False,
                    "temperature": 0.8
                }
            },
            use_examples=True,
            preferred_examples_count=3,
            api_endpoint="https://api.deepseek.com/v1/chat/completions",
            requires_api_key=True,
            supports_streaming=True,
            max_context_length=8192,
            recommended_metrics=["pass@1", "pass@5", "pass@10"]
        ))
        
        # LLaMA/CodeLLaMA系列配置
        self.register_config(ModelConfig(
            name="codellama-instruct",
            family=ModelFamily.LLAMA,
            type=ModelType.INSTRUCT,
            default_temperature=0.8,
            default_max_tokens=1024,
            dataset_preferences={
                "mbpp": {"temperature": 0.7},
                "humaneval": {"temperature": 0.8, "max_tokens": 512},
                "apps": {"temperature": 0.9, "max_tokens": 2048}
            },
            batch_size_recommendation=2,
            max_context_length=4096,
            supports_system_prompt=True
        ))
        
        self.register_config(ModelConfig(
            name="codellama-base", 
            family=ModelFamily.LLAMA,
            type=ModelType.BASE,
            default_temperature=0.7,
            default_max_tokens=512,
            dataset_preferences={
                "humaneval": {"temperature": 0.6, "max_tokens": 256}
            },
            supports_system_prompt=False,
            batch_size_recommendation=4
        ))
        
        # StarCoder系列配置
        self.register_config(ModelConfig(
            name="starcoder-base",
            family=ModelFamily.STARCODER,
            type=ModelType.BASE,
            default_temperature=0.6,
            default_max_tokens=512,
            dataset_preferences={
                "humaneval": {
                    "temperature": 0.5,
                    "use_special_prefix": True,
                    "max_tokens": 256
                },
                "mbpp": {"temperature": 0.7}
            },
            supports_system_prompt=False,
            batch_size_recommendation=4
        ))
        
        self.register_config(ModelConfig(
            name="wizardcoder",
            family=ModelFamily.STARCODER,
            type=ModelType.INSTRUCT,
            default_temperature=0.7,
            default_max_tokens=512,
            dataset_preferences={
                "mbpp": {"temperature": 0.6},
                "apps": {"temperature": 0.8, "max_tokens": 1024}
            },
            supports_system_prompt=True,
            batch_size_recommendation=2
        ))
        
        # Qwen系列配置
        self.register_config(ModelConfig(
            name="qwen-coder",
            family=ModelFamily.QWEN,
            type=ModelType.INSTRUCT,
            default_temperature=0.8,
            default_max_tokens=1024,
            dataset_preferences={
                "mbpp": {"temperature": 0.7},
                "humaneval": {"temperature": 0.8},
                "apps": {"temperature": 0.9, "max_tokens": 2048}
            },
            api_endpoint="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            requires_api_key=True,
            max_context_length=8192
        ))
        
        # OpenAI系列配置
        self.register_config(ModelConfig(
            name="gpt-3.5-turbo",
            family=ModelFamily.OPENAI,
            type=ModelType.CHAT,
            default_temperature=0.7,
            default_max_tokens=512,
            dataset_preferences={
                "mbpp": {"temperature": 0.6, "max_tokens": 256},
                "humaneval": {"temperature": 0.7, "max_tokens": 512},
                "apps": {"temperature": 0.8, "max_tokens": 1024}
            },
            api_endpoint="https://api.openai.com/v1/chat/completions",
            requires_api_key=True,
            supports_streaming=True,
            max_context_length=4096,
            recommended_metrics=["pass@1", "pass@3", "pass@5"]
        ))
        
        self.register_config(ModelConfig(
            name="gpt-4",
            family=ModelFamily.OPENAI,
            type=ModelType.CHAT,
            default_temperature=0.6,
            default_max_tokens=1024,
            dataset_preferences={
                "mbpp": {"temperature": 0.5, "max_tokens": 512},
                "humaneval": {"temperature": 0.6, "max_tokens": 512},
                "apps": {"temperature": 0.7, "max_tokens": 2048}
            },
            api_endpoint="https://api.openai.com/v1/chat/completions",
            requires_api_key=True,
            supports_streaming=True,
            max_context_length=8192,
            batch_size_recommendation=1,  # GPT-4成本较高，建议小批量
            recommended_metrics=["pass@1", "pass@3"]
        ))
        
        # Claude系列配置
        self.register_config(ModelConfig(
            name="claude-3-sonnet",
            family=ModelFamily.CLAUDE,
            type=ModelType.CHAT,
            default_temperature=0.7,
            default_max_tokens=1024,
            dataset_preferences={
                "mbpp": {"temperature": 0.6},
                "humaneval": {"temperature": 0.7},
                "apps": {"temperature": 0.8, "max_tokens": 2048}
            },
            api_endpoint="https://api.anthropic.com/v1/messages",
            requires_api_key=True,
            max_context_length=8192,
            timeout_seconds=60  # Claude可能需要更长时间
        ))
    
    def register_config(self, config: ModelConfig):
        """注册模型配置"""
        key = f"{config.family.value}_{config.type.value}"
        if key not in self.configs:
            self.configs[key] = []
        self.configs[key].append(config)
    
    def get_config_for_model(self, model_name: str) -> ModelConfig:
        """为特定模型获取配置"""
        
        # 首先尝试精确匹配
        for config_list in self.configs.values():
            for config in config_list:
                if config.name.lower() in model_name.lower():
                    return config
        
        # 然后基于模型检测来匹配
        model_info = detect_model_info(model_name)
        key = f"{model_info.family.value}_{model_info.type.value}"
        
        if key in self.configs and self.configs[key]:
            # 返回第一个匹配的配置
            base_config = self.configs[key][0]
            
            # 创建一个针对该模型的定制配置
            custom_config = ModelConfig(
                name=model_name,
                family=model_info.family,
                type=model_info.type,
                default_temperature=base_config.default_temperature,
                default_max_tokens=base_config.default_max_tokens,
                default_top_p=base_config.default_top_p,
                default_num_beams=base_config.default_num_beams,
                dataset_preferences=base_config.dataset_preferences.copy(),
                batch_size_recommendation=base_config.batch_size_recommendation,
                use_examples=base_config.use_examples,
                preferred_examples_count=base_config.preferred_examples_count,
                api_endpoint=base_config.api_endpoint,
                requires_api_key=base_config.requires_api_key,
                supports_streaming=base_config.supports_streaming,
                timeout_seconds=base_config.timeout_seconds,
                needs_special_encoding=base_config.needs_special_encoding,
                supports_system_prompt=base_config.supports_system_prompt,
                max_context_length=base_config.max_context_length,
                recommended_metrics=base_config.recommended_metrics.copy()
            )
            return custom_config
        
        # 返回默认配置
        return self._create_default_config(model_name, model_info)
    
    def _create_default_config(self, model_name: str, model_info) -> ModelConfig:
        """创建默认配置"""
        return ModelConfig(
            name=model_name,
            family=model_info.family,
            type=model_info.type,
            default_temperature=0.8,
            default_max_tokens=512,
            dataset_preferences={},
            batch_size_recommendation=1,
            use_examples=False,
            preferred_examples_count=3,
            requires_api_key=False,
            supports_streaming=False,
            timeout_seconds=30,
            supports_system_prompt=True,
            max_context_length=2048,
            recommended_metrics=["pass@1", "pass@5"]
        )
    
    def get_optimal_params(self, model_name: str, dataset: str) -> Dict[str, Any]:
        """获取模型在特定数据集上的最优参数"""
        config = self.get_config_for_model(model_name)
        
        # 基础参数
        params = {
            "temperature": config.default_temperature,
            "max_tokens": config.default_max_tokens,
            "top_p": config.default_top_p,
            "num_beams": config.default_num_beams,
            "batch_size": config.batch_size_recommendation,
            "timeout": config.timeout_seconds
        }
        
        # 数据集特定优化
        if dataset in config.dataset_preferences:
            dataset_params = config.dataset_preferences[dataset]
            params.update(dataset_params)
        
        # Few-shot examples设置
        if config.use_examples:
            params["use_examples"] = True
            params["examples_count"] = config.preferred_examples_count
        
        return params
    
    def get_model_recommendations(self, dataset: str, 
                                performance_priority: str = "accuracy") -> List[str]:
        """为特定数据集推荐模型"""
        recommendations = []
        
        if dataset.lower() == "mbpp":
            if performance_priority == "accuracy":
                recommendations = [
                    "deepseek-ai/deepseek-coder-6.7b-instruct",  # 高准确率
                    "gpt-4",  # 最高准确率（但成本高）
                    "meta-llama/CodeLlama-7b-Instruct-hf"  # 平衡性能
                ]
            elif performance_priority == "speed":
                recommendations = [
                    "deepseek-ai/deepseek-coder-1.3b-instruct",  # 快速推理
                    "bigcode/starcoder-1b",  # 轻量级
                    "meta-llama/CodeLlama-7b-hf"  # base模型更快
                ]
            elif performance_priority == "cost":
                recommendations = [
                    "deepseek-ai/deepseek-coder-1.3b-instruct",  # 开源免费
                    "gpt-3.5-turbo",  # API成本较低
                    "bigcode/starcoder-7b"  # 开源
                ]
        
        elif dataset.lower() == "humaneval":
            if performance_priority == "accuracy":
                recommendations = [
                    "gpt-4",  # 业界标杆
                    "claude-3-sonnet",  # 强大推理能力
                    "bigcode/starcoder-15b"  # HumanEval优化
                ]
            elif performance_priority == "speed":
                recommendations = [
                    "bigcode/starcoder-7b",  # HumanEval特殊优化
                    "meta-llama/CodeLlama-7b-hf",  # 快速base模型
                    "deepseek-ai/deepseek-coder-1.3b-instruct"
                ]
        
        elif dataset.lower() == "apps":
            if performance_priority == "accuracy":
                recommendations = [
                    "gpt-4",  # 复杂问题处理能力强
                    "claude-3-sonnet",  # 优秀的推理能力
                    "meta-llama/CodeLlama-13b-Instruct-hf"  # 大模型处理复杂问题
                ]
            elif performance_priority == "speed":
                recommendations = [
                    "deepseek-ai/deepseek-coder-6.7b-instruct",
                    "meta-llama/CodeLlama-7b-Instruct-hf",
                    "WizardLM/WizardCoder-15B-V1.0"
                ]
        
        return recommendations
    
    def validate_setup(self, model_name: str) -> Dict[str, Any]:
        """验证模型设置"""
        config = self.get_config_for_model(model_name)
        
        validation = {
            "model_name": model_name,
            "config": config,
            "status": "valid",
            "warnings": [],
            "requirements": [],
            "suggestions": []
        }
        
        # 检查API requirements
        if config.requires_api_key:
            validation["requirements"].append("需要配置API密钥")
            validation["suggestions"].append(f"设置环境变量或配置API key for {config.family.value}")
        
        # 检查上下文长度
        if config.max_context_length < 2048:
            validation["warnings"].append("模型上下文长度较短，可能影响复杂问题的处理")
        
        # 检查特殊处理需求
        if config.needs_special_encoding:
            validation["requirements"].append("需要特殊的编码处理")
        
        return validation


# 全局配置管理器实例
model_config_manager = ModelConfigManager()


def get_model_config(model_name: str) -> ModelConfig:
    """便捷函数：获取模型配置"""
    return model_config_manager.get_config_for_model(model_name)


def get_optimal_generation_params(model_name: str, dataset: str) -> Dict[str, Any]:
    """便捷函数：获取最优生成参数"""
    return model_config_manager.get_optimal_params(model_name, dataset)


def recommend_models(dataset: str, priority: str = "accuracy") -> List[str]:
    """便捷函数：推荐模型"""
    return model_config_manager.get_model_recommendations(dataset, priority)


def validate_model_setup(model_name: str) -> Dict[str, Any]:
    """便捷函数：验证模型设置"""
    return model_config_manager.validate_setup(model_name)


# 使用示例
if __name__ == "__main__":
    print("🔧 智能模型配置系统演示：\n")
    
    test_models = [
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "meta-llama/CodeLlama-7b-Instruct-hf", 
        "bigcode/starcoder-15b",
        "gpt-3.5-turbo"
    ]
    
    for model in test_models:
        print(f"📋 模型: {model}")
        config = get_model_config(model)
        print(f"   家族: {config.family.value}")
        print(f"   类型: {config.type.value}")
        
        # MBPP优化参数
        mbpp_params = get_optimal_generation_params(model, "mbpp")
        print(f"   MBPP最优参数: {mbpp_params}")
        
        # 验证设置
        validation = validate_model_setup(model)
        if validation["warnings"]:
            print(f"   ⚠️  警告: {validation['warnings']}")
        
        print()
    
    # 推荐演示
    print("📊 数据集模型推荐:")
    for dataset in ["mbpp", "humaneval", "apps"]:
        recommendations = recommend_models(dataset, "accuracy")
        print(f"   {dataset.upper()}: {recommendations[:2]}")  # 显示前2个推荐
    
    print("\n✅ 模型配置系统演示完成！") 