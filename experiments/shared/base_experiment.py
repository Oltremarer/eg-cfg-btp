"""
共享的基础实验类，为所有数据集提供通用功能
"""
import os
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class BaseExperiment(ABC):
    """所有实验的基础类"""
    
    def __init__(self, dataset_name: str, model_name: str, config_path: Optional[str] = None):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.config = self.load_config(config_path)
        self.results_dir = f"experiments/{dataset_name}/results"
        self.ensure_results_dir()
    
    @abstractmethod
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载数据集特定配置"""
        pass
    
    @abstractmethod
    def load_dataset(self) -> Dict[str, Any]:
        """加载数据集"""
        pass
    
    @abstractmethod  
    def format_prompt(self, problem: Dict[str, Any]) -> str:
        """格式化提示词"""
        pass
    
    def ensure_results_dir(self):
        """确保结果目录存在"""
        os.makedirs(self.results_dir, exist_ok=True)
    
    def save_results(self, results: Dict, experiment_type: str, suffix: str = ""):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix:
            filename = f"{experiment_type}_{suffix}_{timestamp}.json"
        else:
            filename = f"{experiment_type}_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已保存到: {filepath}")
        return filepath
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """获取实验配置信息"""
        return {
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'dataset_config': self.config
        }


class DatasetExperiment(BaseExperiment):
    """数据集特定实验的基类"""
    
    def __init__(self, dataset_name: str, model_name: str, use_openai: bool = False):
        self.use_openai = use_openai
        super().__init__(dataset_name, model_name)
        self.problems = self.load_dataset()
    
    def run_on_problem_subset(self, max_problems: Optional[int] = None) -> List[tuple]:
        """获取问题子集用于实验"""
        problems_list = list(self.problems.items())
        if max_problems:
            problems_list = problems_list[:max_problems]
        return problems_list
    
    def calculate_pass_at_k(self, results: Dict[str, Any], k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """计算Pass@k指标"""
        if not results:
            return {f"pass@{k}": 0.0 for k in k_values}
        
        pass_at_k = {}
        
        for k in k_values:
            successes = 0
            total = 0
            
            for task_id, result in results.items():
                if isinstance(result, dict) and 'solutions' in result:
                    solutions = result['solutions']
                    total += 1
                    
                    # 检查前k个解决方案中是否有完全通过的
                    passed = False
                    for i, solution in enumerate(solutions[:k]):
                        if solution.get('fully_passed', False):
                            passed = True
                            break
                    
                    if passed:
                        successes += 1
            
            pass_at_k[f"pass@{k}"] = successes / total if total > 0 else 0.0
        
        return pass_at_k


class Step1BaselineExperiment(DatasetExperiment):
    """Step1 基线实验基类"""
    
    def run_experiment(self, num_samples: int = 10, temperature: float = 0.8, 
                      max_problems: Optional[int] = None) -> Dict[str, Any]:
        """运行基线实验的通用框架"""
        problems_list = self.run_on_problem_subset(max_problems)
        results = {}
        
        print(f"开始运行基线实验，共 {len(problems_list)} 个问题")
        
        for task_id, problem in problems_list:
            print(f"处理问题 {task_id}...")
            
            try:
                solutions = self.generate_solutions(problem, num_samples, temperature)
                tested_solutions = self.test_solutions(problem, solutions)
                
                results[str(task_id)] = {
                    'problem': problem,
                    'solutions': tested_solutions
                }
                
            except Exception as e:
                print(f"问题 {task_id} 处理失败: {e}")
                results[str(task_id)] = {
                    'problem': problem,
                    'error': str(e),
                    'solutions': []
                }
        
        return results
    
    @abstractmethod
    def generate_solutions(self, problem: Dict[str, Any], num_samples: int, 
                          temperature: float) -> List[str]:
        """生成解决方案"""
        pass
    
    @abstractmethod
    def test_solutions(self, problem: Dict[str, Any], 
                      solutions: List[str]) -> List[Dict[str, Any]]:
        """测试解决方案"""
        pass


class Step2BTPExperiment(DatasetExperiment):
    """Step2 BTP实验基类"""
    
    def run_experiment(self, max_problems: int = 100, num_beams: int = 5,
                      n_iterations: int = 3, batch_size: int = 100) -> Dict[str, Any]:
        """运行BTP实验的通用框架"""
        problems_list = self.run_on_problem_subset(max_problems)
        
        print(f"开始运行BTP实验，共 {len(problems_list)} 个问题")
        
        # 阶段1: Beam Search + Testing
        self.phase1_beam_search_sampling(problems_list, num_beams)
        
        # 阶段2: Prioritized Experience Replay
        self.phase2_pper_training(n_iterations, batch_size)
        
        return self.get_experiment_results()
    
    @abstractmethod
    def phase1_beam_search_sampling(self, problems_list: List[tuple], num_beams: int):
        """阶段1: 束搜索采样"""
        pass
    
    @abstractmethod
    def phase2_pper_training(self, n_iterations: int, batch_size: int):
        """阶段2: 优先经验回放训练"""
        pass
    
    @abstractmethod
    def get_experiment_results(self) -> Dict[str, Any]:
        """获取实验结果"""
        pass 