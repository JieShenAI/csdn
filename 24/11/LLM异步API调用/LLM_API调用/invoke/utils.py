import random
import time
from typing import Union, Dict


def generate_arithmetic_expression(num: int):
    """
    num: 几个操作符
    """
    # 定义操作符和数字范围，除法
    operators = ['+', '-', '*']
    expression = f"{random.randint(1, 100)} {random.choice(operators)} {random.randint(1, 100)}"
    num -= 1
    for _ in range(num):
        expression = f"{expression} {random.choice(operators)} {random.randint(1, 100)}"
    result = eval(expression)
    return expression, result


import json
import re

# 假设这是模型返回的内容
model_response = '''
Here is your calculation:
{
  "expression": "70 + 81 - 43 + 48 + 9",
  "result": 165
}
'''


def re_parse_json(text) -> Union[Dict, None]:
    # 提取 JSON 内容
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        json_data = json_match.group(0)
        response_data = json.loads(json_data)
        return response_data
    print(f"异常:\n{text}")
    return None


def calculate_time_difference(start_time, end_time):
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = (elapsed_time - int(elapsed_time)) * 1000

    print(
        f"executed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03} (h:m:s.ms)"
    )

def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间

        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        milliseconds = (elapsed_time - int(elapsed_time)) * 1000

        print(
            f"Function '{func.__name__}' executed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03} (h:m:s.ms)")
        return result

    return wrapper

# 测试生成
if __name__ == "__main__":
    expr, res = generate_arithmetic_expression(4)
    print(f"生成的运算表达式: {expr}")
    print(f"计算结果: {res}")
