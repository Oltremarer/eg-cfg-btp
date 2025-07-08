"""
数据集特定配置文件
"""
from typing import Dict, Any


# MBPP数据集配置
MBPP_CONFIG = {
    "name": "mbpp",
    "description": "Mostly Basic Python Problems",
    "data_path": "data/mbpp",
    "prompt_template": "english",  # 使用英文提示
    "test_timeout": 10.0,  # 测试超时时间（秒）
    "max_code_length": 2048,  # 最大代码长度
    "supported_languages": ["python"],
    "evaluation_metrics": ["pass@1", "pass@5", "pass@10"],
    "default_params": {
        "num_samples": 10,
        "temperature": 0.8,
        "max_problems": 50
    }
}

# HumanEval数据集配置
HUMANEVAL_CONFIG = {
    "name": "humaneval",
    "description": "Human Eval Code Generation Benchmark",
    "data_path": "data/humaneval",
    "prompt_template": "english",
    "test_timeout": 10.0,
    "max_code_length": 2048,
    "supported_languages": ["python"],
    "evaluation_metrics": ["pass@1", "pass@5", "pass@10"],
    "default_params": {
        "num_samples": 10,
        "temperature": 0.8,
        "max_problems": 164  # HumanEval共164个问题
    }
}

# APPS数据集配置
APPS_CONFIG = {
    "name": "apps",
    "description": "APPS Programming Problems",
    "data_path": "data/apps",
    "prompt_template": "english",
    "test_timeout": 15.0,  # APPS问题通常更复杂，给更多时间
    "max_code_length": 4096,  # APPS可能需要更长的代码
    "supported_languages": ["python"],
    "evaluation_metrics": ["pass@1", "pass@5", "pass@10"],
    "default_params": {
        "num_samples": 10,
        "temperature": 0.8,
        "max_problems": 100
    },
    "difficulty_levels": ["introductory", "interview", "competition"]
}

# 统一配置映射
DATASET_CONFIGS = {
    "mbpp": MBPP_CONFIG,
    "humaneval": HUMANEVAL_CONFIG,
    "apps": APPS_CONFIG
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """获取指定数据集的配置"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集: {dataset_name}. 支持的数据集: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_name].copy()


def get_supported_datasets() -> list:
    """获取支持的数据集列表"""
    return list(DATASET_CONFIGS.keys())


def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """验证数据集配置是否有效"""
    required_fields = [
        "name", "description", "data_path", "prompt_template",
        "test_timeout", "max_code_length", "supported_languages",
        "evaluation_metrics", "default_params"
    ]
    
    for field in required_fields:
        if field not in config:
            print(f"缺少必需字段: {field}")
            return False
    
    return True


# 英文提示模板（避免中文导致的问题）
ENGLISH_PROMPT_TEMPLATES = {
    "mbpp": {
        "system": "You are a helpful coding assistant. Generate Python code to solve the given problem.",
        "user_template": "Problem: {text}\n\nPlease write a Python function to solve this problem. Only provide the function definition, no additional explanation.",
        "completion_prefix": "def "
    },
    
    "humaneval": {
        "system": "You are a helpful coding assistant. Complete the given Python function.",
        "user_template": "{prompt}",
        "completion_prefix": ""
    },
    
    "apps": {
        "system": "You are a helpful coding assistant. Solve the programming problem step by step.",
        "user_template": "Problem:\n{problem_description}\n\nPlease provide a complete Python solution.",
        "completion_prefix": ""
    }
}


def get_prompt_template(dataset_name: str) -> Dict[str, str]:
    """获取数据集的提示模板"""
    if dataset_name not in ENGLISH_PROMPT_TEMPLATES:
        raise ValueError(f"没有为数据集 {dataset_name} 定义提示模板")
    
    return ENGLISH_PROMPT_TEMPLATES[dataset_name].copy() 