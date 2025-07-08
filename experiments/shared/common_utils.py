"""
通用工具函数
"""
import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def safe_import_eg_cfg():
    """安全导入eg_cfg模块"""
    try:
        import eg_cfg
        return eg_cfg
    except ImportError as e:
        print(f"警告: 无法导入eg_cfg模块: {e}")
        return None


def load_mbpp_problems() -> Dict[str, Any]:
    """加载MBPP数据集"""
    try:
        eg_cfg = safe_import_eg_cfg()
        if eg_cfg is None:
            raise ImportError("eg_cfg模块不可用")
        
        from eg_cfg.mbpp_utils import load_mbpp_problems as _load_mbpp_problems
        return _load_mbpp_problems()
    except Exception as e:
        print(f"加载MBPP数据集失败: {e}")
        return {}


def load_humaneval_problems() -> Dict[str, Any]:
    """加载HumanEval数据集"""
    try:
        # 这里可以添加HumanEval数据集加载逻辑
        # 目前返回空字典，待具体实现
        print("HumanEval数据集加载功能待实现")
        return {}
    except Exception as e:
        print(f"加载HumanEval数据集失败: {e}")
        return {}


def load_apps_problems() -> Dict[str, Any]:
    """加载APPS数据集"""
    try:
        # 这里可以添加APPS数据集加载逻辑
        # 目前返回空字典，待具体实现
        print("APPS数据集加载功能待实现")
        return {}
    except Exception as e:
        print(f"加载APPS数据集失败: {e}")
        return {}


def get_dataset_loader(dataset_name: str):
    """获取数据集加载函数"""
    loaders = {
        "mbpp": load_mbpp_problems,
        "humaneval": load_humaneval_problems,
        "apps": load_apps_problems
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return loaders[dataset_name]


def safe_execute_code(code: str, test_cases: List[str], timeout: float = 10.0) -> Dict[str, Any]:
    """安全执行代码并运行测试用例"""
    results = {
        "passed_tests": 0,
        "total_tests": len(test_cases),
        "pass_rate": 0.0,
        "fully_passed": False,
        "test_results": {},
        "execution_error": None
    }
    
    try:
        # 创建执行环境
        exec_globals = {"__builtins__": __builtins__}
        
        # 执行代码
        exec(code, exec_globals)
        
        # 运行测试用例
        for i, test_case in enumerate(test_cases):
            try:
                start_time = time.time()
                exec(test_case, exec_globals)
                execution_time = time.time() - start_time
                
                results["test_results"][test_case] = {
                    "result": True,
                    "time": execution_time,
                    "error": None
                }
                results["passed_tests"] += 1
                
            except Exception as e:
                results["test_results"][test_case] = {
                    "result": False,
                    "time": 0.0,
                    "error": str(e)
                }
        
        # 计算通过率
        if results["total_tests"] > 0:
            results["pass_rate"] = results["passed_tests"] / results["total_tests"]
            results["fully_passed"] = (results["passed_tests"] == results["total_tests"])
    
    except Exception as e:
        results["execution_error"] = str(e)
        results["pass_rate"] = 0.0
        results["fully_passed"] = False
    
    return results


def format_experiment_summary(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """格式化实验结果摘要"""
    summary = []
    summary.append("=" * 60)
    summary.append("实验结果摘要")
    summary.append("=" * 60)
    
    # 实验配置
    summary.append(f"数据集: {config.get('dataset_name', 'Unknown')}")
    summary.append(f"模型: {config.get('model_name', 'Unknown')}")
    summary.append(f"时间: {config.get('timestamp', 'Unknown')}")
    summary.append("")
    
    # 结果统计
    if 'metrics' in results:
        summary.append("Pass@k 指标:")
        for metric, value in results['metrics'].items():
            summary.append(f"  {metric}: {value:.3f}")
        summary.append("")
    
    # 问题统计
    if 'results' in results:
        problem_count = len(results['results'])
        successful_problems = sum(1 for r in results['results'].values() 
                                if isinstance(r, dict) and r.get('solutions') and 
                                any(s.get('fully_passed', False) for s in r.get('solutions', [])))
        
        summary.append(f"问题总数: {problem_count}")
        summary.append(f"成功解决: {successful_problems}")
        summary.append(f"成功率: {successful_problems/problem_count:.1%}" if problem_count > 0 else "成功率: 0%")
    
    summary.append("=" * 60)
    return "\n".join(summary)


def ensure_directory(path: str) -> Path:
    """确保目录存在"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json_with_backup(data: Dict[str, Any], filepath: str) -> bool:
    """保存JSON文件，如果文件存在则创建备份"""
    file_path = Path(filepath)
    
    try:
        # 如果文件存在，创建备份
        if file_path.exists():
            backup_path = file_path.with_suffix(f'.backup_{int(time.time())}.json')
            file_path.rename(backup_path)
            print(f"原文件已备份到: {backup_path}")
        
        # 保存新文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"文件已保存到: {file_path}")
        return True
        
    except Exception as e:
        print(f"保存文件失败: {e}")
        return False


def log_error_with_context(error: Exception, context: Dict[str, Any], 
                          log_dir: str = "experiments/logs") -> str:
    """记录错误及其上下文"""
    ensure_directory(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"error_{timestamp}.log"
    
    error_info = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "context": context
    }
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        
        print(f"错误日志已保存到: {log_file}")
        return str(log_file)
        
    except Exception as log_error:
        print(f"保存错误日志失败: {log_error}")
        return ""


def validate_model_name(model_name: str) -> bool:
    """验证模型名称是否有效"""
    if not model_name or not isinstance(model_name, str):
        return False
    
    # 基本格式检查
    if len(model_name.strip()) == 0:
        return False
    
    return True


def get_experiment_timestamp() -> str:
    """获取实验时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """打印进度信息"""
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"\r{prefix}: {current}/{total} ({percentage:.1f}%)", end="", flush=True)
    if current == total:
        print()  # 换行 