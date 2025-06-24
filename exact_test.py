import subprocess
import tempfile

# 从JSON文件中精确复制的代码和测试用例
code = """def sort_matrix(matrix):
    return sorted(matrix, key=sum)"""

test_case = """assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]"""

# 完整的测试代码
full_code = f"{code}\n{test_case}"

print("=== 精确测试 ===")
print("代码:")
print(repr(code))
print("\n测试用例:")
print(repr(test_case))
print("\n完整代码:")
print(repr(full_code))
print("\n执行结果:")

# 写入文件并执行
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
    f.write(full_code)
    f.flush()
    
    result = subprocess.run(
        ['python', f.name], 
        capture_output=True, 
        text=True,
        encoding='utf-8'
    )
    
    print(f"返回码: {result.returncode}")
    print(f"stdout: {repr(result.stdout)}")
    print(f"stderr: {repr(result.stderr)}")
    
    # 按照框架逻辑判断
    test_passed = result.returncode == 0 and not "Traceback" in result.stderr
    print(f"框架判断: {'✅ 通过' if test_passed else '❌ 失败'}")

print("\n=== 手动验证 ===")
# 手动执行相同的逻辑
def sort_matrix(matrix):
    return sorted(matrix, key=sum)

actual = sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])
expected = [[1, 1, 1], [1, 2, 3], [2, 4, 5]]
print(f"实际结果: {actual}")
print(f"期望结果: {expected}")
print(f"手动验证: {'✅ 相等' if actual == expected else '❌ 不等'}") 