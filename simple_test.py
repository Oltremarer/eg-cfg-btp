#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 直接模拟框架的行为
import subprocess
import tempfile

# 从JSON文件复制的实际代码和测试
code = """def sort_matrix(matrix):
    return sorted(matrix, key=sum)"""

test_case = """assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]"""

# 组合代码
full_code = code + "\n" + test_case

print("代码内容:")
print(repr(full_code))
print("\n执行测试...")

# 写入临时文件并执行
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(full_code)
    f.flush()
    
    # 执行
    result = subprocess.run(['python', f.name], capture_output=True, text=True)
    
    print(f"返回码: {result.returncode}")
    print(f"stdout: {repr(result.stdout)}")
    print(f"stderr: {repr(result.stderr)}")
    
    if result.returncode == 0:
        print("✅ 测试通过!")
    else:
        print("❌ 测试失败!") 