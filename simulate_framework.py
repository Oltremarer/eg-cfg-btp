import subprocess
import tempfile
import time

def evaluate_solution(code, test_case, timeout=10):
    """完全复制框架中的 evaluate_solution 函数"""
    test_passed = False
    error = None
    test_code = f"{code}\n{test_case}"
    
    print(f"完整测试代码:")
    print("=" * 50)
    print(test_code)
    print("=" * 50)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
        temp_file.write(test_code)
        temp_file.flush()

        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", temp_file.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            print(f"返回码: {result.returncode}")
            print(f"stdout: '{result.stdout}'")
            print(f"stderr: '{result.stderr}'")
            print(f"包含Traceback: {'Traceback' in result.stderr}")

            if result.returncode == 0 and not "Traceback" in result.stderr:
                test_passed = True

        except subprocess.TimeoutExpired:
            error = "Timeout"
            pass
        except Exception as e:
            error = "Exception"
            pass
        finally:
            end_time = time.time()
            delta_time = end_time - start_time

    result_entry = {"result": test_passed, "time": delta_time, "error": error}
    return result_entry

# 使用JSON文件中实际的代码和测试用例
print("=== 测试矩阵排序 (Task 12) ===")
code = "def sort_matrix(matrix):\n    return sorted(matrix, key=sum)"
test_case = "assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]"

result = evaluate_solution(code, test_case)
print(f"测试结果: {result}")
print()

print("=== 测试移除字符 (Task 11) ===")
code2 = "def remove_Occ(s, char):\n    first = s.find(char)\n    last = s.rfind(char)\n    if first != -1 and last != -1:\n        s = s[:first] + s[first+1:last] + s[last+1:]\n    return s"
test_case2 = 'assert remove_Occ("hello","l") == "heo"'

result2 = evaluate_solution(code2, test_case2)
print(f"测试结果: {result2}") 