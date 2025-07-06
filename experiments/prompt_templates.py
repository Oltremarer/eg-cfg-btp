"""
统一的提示词模板管理系统
确保所有实验使用一致的英文提示词格式
"""

class PromptTemplates:
    """统一的提示词模板管理"""
    
    # MBPP数据集的提示词模板
    MBPP_TEMPLATE = """Solve the following programming problem:

Problem: {problem_text}

Test cases:
{test_cases}

Provide a complete Python function:

```python
"""
    
    # HumanEval数据集的提示词模板
    HUMANEVAL_TEMPLATE = """Complete the following Python function:

{problem_prompt}

```python
"""
    
    # APPS数据集的提示词模板
    APPS_TEMPLATE = """Solve the following programming problem:

Problem: {problem_text}

Test cases:
{test_cases}

Provide a complete Python function:

```python
"""
    
    # 微调训练的指令模板
    FINETUNE_INSTRUCTION_TEMPLATE = """### Instruction:
Solve this programming problem:
{problem_text}

### Response:
{code_solution}"""
    
    @classmethod
    def get_prompt(cls, dataset: str, problem: dict) -> str:
        """
        根据数据集类型和问题获取格式化的提示词
        
        Args:
            dataset: 数据集名称 ('mbpp', 'humaneval', 'apps')
            problem: 问题字典
            
        Returns:
            格式化的提示词字符串
        """
        if dataset.lower() == "mbpp":
            return cls.MBPP_TEMPLATE.format(
                problem_text=problem['text'],
                test_cases='\n'.join(problem['test_list'])
            )
        elif dataset.lower() == "humaneval":
            return cls.HUMANEVAL_TEMPLATE.format(
                problem_prompt=problem['prompt']
            )
        elif dataset.lower() == "apps":
            return cls.APPS_TEMPLATE.format(
                problem_text=problem['question'],
                test_cases='\n'.join(problem['test_list'])
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    
    @classmethod
    def get_finetune_prompt(cls, problem_text: str, code_solution: str) -> str:
        """
        获取微调训练的指令格式提示词
        
        Args:
            problem_text: 问题描述
            code_solution: 代码解决方案
            
        Returns:
            微调训练格式的提示词
        """
        return cls.FINETUNE_INSTRUCTION_TEMPLATE.format(
            problem_text=problem_text,
            code_solution=code_solution
        )
    
    @classmethod
    def validate_consistency(cls) -> bool:
        """
        验证不同模板的一致性
        
        Returns:
            是否一致
        """
        # 检查所有模板都使用英文
        english_keywords = ["Solve", "Problem", "Test cases", "Provide", "Complete"]
        chinese_keywords = ["解决", "问题", "测试用例", "请", "编写"]
        
        templates = [cls.MBPP_TEMPLATE, cls.HUMANEVAL_TEMPLATE, cls.APPS_TEMPLATE]
        
        for template in templates:
            # 检查是否包含中文关键词
            if any(keyword in template for keyword in chinese_keywords):
                print(f"❌ Template contains Chinese keywords: {template[:50]}...")
                return False
            
            # 检查是否包含英文关键词
            if not any(keyword in template for keyword in english_keywords):
                print(f"❌ Template missing English keywords: {template[:50]}...")
                return False
        
        print("✅ All templates are consistent and use English")
        return True


# 使用示例
if __name__ == "__main__":
    # 验证模板一致性
    PromptTemplates.validate_consistency()
    
    # 示例问题
    mbpp_problem = {
        'text': 'Write a function to find the maximum element in a list.',
        'test_list': ['assert max_element([1, 2, 3]) == 3', 'assert max_element([10, 5, 8]) == 10']
    }
    
    # 获取提示词
    prompt = PromptTemplates.get_prompt("mbpp", mbpp_problem)
    print("Generated prompt:")
    print(prompt) 