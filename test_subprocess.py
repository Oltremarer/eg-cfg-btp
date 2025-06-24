import subprocess
import tempfile

# 测试 1: 成功的 assert
code1 = """def sort_matrix(matrix):
    return sorted(matrix, key=sum)

assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]]) == [[1, 1, 1], [1, 2, 3], [2, 4, 5]]
"""

# 测试 2: 失败的 assert  
code2 = """def sort_matrix(matrix):
    return sorted(matrix, key=sum)

assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]]) == [[1, 1, 1], [1, 2, 3], [2, 4, 5], [999]]
"""

def test_code(code, description):
    print(f"\n=== {description} ===")
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
        temp_file.write(code)
        temp_file.flush()
        
        result = subprocess.run(
            ["python", temp_file.name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        print(f"返回码: {result.returncode}")
        print(f"stdout: '{result.stdout}'")
        print(f"stderr: '{result.stderr}'")
        print(f"是否包含 Traceback: {'Traceback' in result.stderr}")
        
        # 原始逻辑
        test_passed_original = result.returncode == 0 and not "Traceback" in result.stderr
        print(f"原始逻辑判断为通过: {test_passed_original}")

test_code(code1, "成功的测试")
test_code(code2, "失败的测试") 