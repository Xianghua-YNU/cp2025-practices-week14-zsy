#!/usr/bin/env python3
"""
GitHub Classroom 自动评分脚本 - 数值微分和积分实验
"""

import os
import sys
import json
import subprocess
import pytest
from pathlib import Path

# 更新测试配置以匹配常微分方程项目结构
TESTS = [
    {"name": "项目1: Lotka-Volterra模型", 
     "file": "project_1_lotka_volterra/tests/test_lotka_volterra.py", 
     "points": 10},
    {"name": "项目2: Lorenz方程系统", 
     "file": "project_2_lorenz_equations/tests/test_lorenz_equations.py", 
     "points": 10},
    {"name": "项目3: 受迫单摆系统", 
     "file": "project_3_forced_pendulum/tests/test_forced_pendulum.py", 
     "points": 10},
    {"name": "项目4: 谐振子模型", 
     "file": "project_4_harmonic_oscillator/tests/test_harmonic_oscillator.py", 
     "points": 10},
    {"name": "项目5: Van der Pol振荡器", 
     "file": "project_5_van_der_pol/tests/test_van_der_pol.py", 
     "points": 10}
]

def run_test(test_file):
    """运行单个测试文件并返回结果"""
    result = pytest.main(["-v", test_file])
    return result == 0  # 0表示测试通过

def calculate_score():
    """计算总分并生成结果报告"""
    total_points = 0
    max_points = 0
    results = []

    for test in TESTS:
        max_points += test["points"]
        test_file = test["file"]
        test_name = test["name"]
        points = test["points"]
        
        print(f"运行测试: {test_name}")
        passed = run_test(test_file)
        
        if passed:
            total_points += points
            status = "通过"
        else:
            status = "失败"
        
        results.append({
            "name": test_name,
            "status": status,
            "points": points if passed else 0,
            "max_points": points
        })
        
        print(f"  状态: {status}")
        print(f"  得分: {points if passed else 0}/{points}")
        print()
    
    # 生成总结
    print(f"总分: {total_points}/{max_points}")
    
    # 生成GitHub Actions兼容的输出
    with open(os.environ.get('GITHUB_STEP_SUMMARY', 'score_summary.md'), 'w') as f:
        f.write("# 自动评分结果\n\n")
        f.write("| 测试 | 状态 | 得分 |\n")
        f.write("|------|------|------|\n")
        
        for result in results:
            f.write(f"| {result['name']} | {result['status']} | {result['points']}/{result['max_points']} |\n")
        
        f.write(f"\n## 总分: {total_points}/{max_points}\n")
    
    # 生成分数JSON文件
    score_data = {
        "score": total_points,
        "max_score": max_points,
        "tests": results
    }
    
    with open('score.json', 'w') as f:
        json.dump(score_data, f, indent=2)
    
    return total_points, max_points

if __name__ == "__main__":
    # 确保工作目录是项目根目录
    os.chdir(Path(__file__).parent.parent.parent)
    
    # 安装依赖
    print("安装依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # 运行测试并计算分数
    print("\n开始评分...\n")
    total, maximum = calculate_score()
    
    # 设置GitHub Actions输出变量
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"points={total}\n")
    
    # 退出代码
    sys.exit(0 if total == maximum else 1)
