# 调试测试脚本 - 验证 GPT-4 生成的代码

# Test 1: 矩阵排序
print("=== Test 1: 矩阵排序 ===")
def sort_matrix(matrix):
    return sorted(matrix, key=sum)

result = sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])
expected = [[1, 1, 1], [1, 2, 3], [2, 4, 5]]
print(f"实际结果: {result}")
print(f"期望结果: {expected}")
print(f"是否相等: {result == expected}")
print()

# Test 2: 字符串移除字符
print("=== Test 2: 移除首尾字符 ===")
def remove_Occ(s, char):
    first = s.find(char)
    last = s.rfind(char)
    if first != -1 and last != -1:
        s = s[:first] + s[first+1:last] + s[last+1:]
    return s

test_cases = [
    ("hello", "l", "heo"),
    ("abcda", "a", "bcd"),
    ("PHP", "P", "H")
]

for s, char, expected in test_cases:
    result = remove_Occ(s, char)
    print(f"remove_Occ('{s}', '{char}') = '{result}', 期望: '{expected}', 正确: {result == expected}")
print()

# Test 3: 计数最常见单词
print("=== Test 3: 最常见单词 ===")
from collections import Counter

def count_common(words):
    return Counter(words).most_common(4)  # 修改：只返回前4个

words = ['red','green','black','pink','black','white','black','eyes','white','black','orange','pink','pink','red','red','white','orange','white',"black",'pink','green','green','pink','green','pink','white','orange',"orange",'red']
result = count_common(words)
expected = [('pink', 6), ('black', 5), ('white', 5), ('red', 4)]
print(f"实际结果: {result}")
print(f"期望结果: {expected}")
print(f"是否相等: {result == expected}")
print()

# Test 4: 三角柱体积
print("=== Test 4: 三角柱体积 ===")
def find_Volume(l, b, h):
    return (l * b * h) // 2  # 修改：使用整数除法

test_cases = [
    (10, 8, 6, 240),
    (3, 2, 2, 6),
    (1, 2, 1, 1)
]

for l, b, h, expected in test_cases:
    result = find_Volume(l, b, h)
    print(f"find_Volume({l}, {b}, {h}) = {result}, 期望: {expected}, 正确: {result == expected}") 