#!/usr/bin/env python3
"""
简单测试脚本：验证DeepSeek 1.3B模型在CPU上的运行
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_cpu_model():
    print("=== DeepSeek 1.3B CPU测试 ===")
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print("警告: 检测到CUDA，但我们将强制使用CPU进行测试")
        device = torch.device("cpu")
        print(f"强制设备: {device}")
    
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    
    print(f"\n正在加载模型: {model_name}")
    print("这可能需要几分钟时间，特别是首次下载...")
    
    try:
        # 加载tokenizer
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型到CPU
        print("加载模型到CPU...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,  # CPU上使用float32
            device_map="cpu",           # 强制使用CPU
            trust_remote_code=True
        )
        load_time = time.time() - start_time
        print(f"模型加载完成，用时: {load_time:.2f}秒")
        
        # 简单的代码生成测试
        prompt = """
Write a Python function to remove the first occurrence of a character from a string.

def remove_first_occurrence(s, char):
"""
        
        print(f"\n测试提示词: {prompt}")
        print("正在生成代码...")
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 生成代码
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n生成用时: {generation_time:.2f}秒")
        print("=" * 50)
        print("生成的代码:")
        print(generated_text)
        print("=" * 50)
        
        print("\n✅ CPU模型测试成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cpu_model() 