#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：Lotka-Volterra捕食者-猎物模型 - 测试代码

测试内容：
1. 基本函数功能测试
2. 数值方法精度测试
3. 边界条件测试
4. 参数敏感性测试
"""

import unittest
import numpy as np
import sys
import os

# 添加项目路径以导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solution.lotka_volterra_solution import (
from lotka_volterra_student import (
    lotka_volterra_system,
    euler_method,
    improved_euler_method,
    runge_kutta_4,
    solve_lotka_volterra,
    compare_methods
)



class TestLotkaVolterraSystem(unittest.TestCase):
    """测试Lotka-Volterra方程组函数"""
    
    def setUp(self):
        """设置测试参数"""
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 0.5
        self.delta = 2.0
        self.tolerance = 1e-10
    
    def test_system_basic_functionality_points_5(self):
        """测试方程组基本功能 - 5分"""
        state = np.array([2.0, 2.0])
        t = 0.0
        
        result = lotka_volterra_system(state, t, self.alpha, self.beta, 
                                     self.gamma, self.delta)
        
        # 验证返回类型和形状
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))
        
        # 验证计算结果
        expected_dxdt = self.alpha * 2.0 - self.beta * 2.0 * 2.0  # 1*2 - 0.5*2*2 = 0
        expected_dydt = self.gamma * 2.0 * 2.0 - self.delta * 2.0  # 0.5*2*2 - 2*2 = -2
        
        self.assertAlmostEqual(result[0], expected_dxdt, places=10)
        self.assertAlmostEqual(result[1], expected_dydt, places=10)
    
    def test_system_equilibrium_point_points_3(self):
        """测试平衡点 - 3分"""
        # 平衡点：x = delta/gamma, y = alpha/beta
        x_eq = self.delta / self.gamma  # 2/0.5 = 4
        y_eq = self.alpha / self.beta   # 1/0.5 = 2
        
        state = np.array([x_eq, y_eq])
        result = lotka_volterra_system(state, 0.0, self.alpha, self.beta, 
                                     self.gamma, self.delta)
        
        # 在平衡点，导数应该为0
        self.assertAlmostEqual(result[0], 0.0, places=10)
        self.assertAlmostEqual(result[1], 0.0, places=10)
    
    def test_system_different_states_points_2(self):
        """测试不同状态下的方程组 - 2分"""
        test_cases = [
            ([1.0, 1.0], [0.5, -1.5]),  # 修改这里的期望值
            ([3.0, 1.0], [1.5, -0.5]),
            ([1.0, 3.0], [-0.5, -4.5])
        ]
        
        for state, expected in test_cases:
            with self.subTest(state=state):
                result = lotka_volterra_system(np.array(state), 0.0, 
                                             self.alpha, self.beta, 
                                             self.gamma, self.delta)
                np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestNumericalMethods(unittest.TestCase):
    """测试数值方法"""
    
    def setUp(self):
        """设置测试参数"""
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 0.5
        self.delta = 2.0
        
        def test_ode(y, t):
            """简单的测试ODE: dy/dt = -y, 解析解为 y = y0 * exp(-t)"""
            return -y
        
        self.test_ode = test_ode
        self.y0_simple = np.array([1.0])
        self.t_span_simple = (0, 1)
        self.dt_simple = 0.1
    
    def test_euler_method_simple_ode_points_5(self):
        """测试欧拉法求解简单ODE - 5分"""
        t, y = euler_method(self.test_ode, self.y0_simple, 
                           self.t_span_simple, self.dt_simple)
        
        # 验证返回格式
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.shape[1], 1)
        
        # 验证初始条件
        self.assertAlmostEqual(y[0, 0], 1.0, places=10)
        
        # 验证最终值（欧拉法的近似解）
        analytical_final = np.exp(-1.0)  # 约0.368
        self.assertLess(abs(y[-1, 0] - analytical_final), 0.1)  # 允许较大误差
    
    def test_improved_euler_method_points_5(self):
        """测试改进欧拉法 - 5分"""
        t, y = improved_euler_method(self.test_ode, self.y0_simple, 
                                   self.t_span_simple, self.dt_simple)
        
        # 验证返回格式
        self.assertEqual(y.shape[1], 1)
        self.assertAlmostEqual(y[0, 0], 1.0, places=10)
        
        # 改进欧拉法应该比欧拉法更精确
        analytical_final = np.exp(-1.0)
        self.assertLess(abs(y[-1, 0] - analytical_final), 0.05)
    
    def test_runge_kutta_4_method_points_8(self):
        """测试4阶龙格-库塔法 - 8分"""
        t, y = runge_kutta_4(self.test_ode, self.y0_simple, 
                           self.t_span_simple, self.dt_simple)
        
        # 验证返回格式
        self.assertEqual(y.shape[1], 1)
        self.assertAlmostEqual(y[0, 0], 1.0, places=10)
        
        # RK4应该非常精确
        analytical_final = np.exp(-1.0)
        self.assertLess(abs(y[-1, 0] - analytical_final), 0.01)
    
    def test_method_convergence_points_7(self):
        """测试方法收敛性 - 7分"""
        dt_values = [0.1, 0.05, 0.025]
        errors_euler = []
        errors_rk4 = []
        
        analytical_final = np.exp(-1.0)
        
        for dt in dt_values:
            # 欧拉法
            t, y_euler = euler_method(self.test_ode, self.y0_simple, 
                                    self.t_span_simple, dt)
            error_euler = abs(y_euler[-1, 0] - analytical_final)
            errors_euler.append(error_euler)
            
            # RK4
            t, y_rk4 = runge_kutta_4(self.test_ode, self.y0_simple, 
                                   self.t_span_simple, dt)
            error_rk4 = abs(y_rk4[-1, 0] - analytical_final)
            errors_rk4.append(error_rk4)
        
        # 验证误差随步长减小而减小
        for i in range(len(dt_values) - 1):
            self.assertLess(errors_euler[i+1], errors_euler[i])
            self.assertLess(errors_rk4[i+1], errors_rk4[i])


class TestLotkaVolterraSolver(unittest.TestCase):
    """测试Lotka-Volterra求解器"""
    
    def setUp(self):
        """设置测试参数"""
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 0.5
        self.delta = 2.0
        self.x0 = 2.0
        self.y0 = 2.0
        self.t_span = (0, 5)
        self.dt = 0.01
    
    def test_solve_basic_functionality_points_8(self):
        """测试基本求解功能 - 8分"""
        t, x, y = solve_lotka_volterra(self.alpha, self.beta, self.gamma, 
                                     self.delta, self.x0, self.y0, 
                                     self.t_span, self.dt)
        
        # 验证返回格式
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(t), len(x))
        self.assertEqual(len(t), len(y))
        
        # 验证初始条件
        self.assertAlmostEqual(x[0], self.x0, places=10)
        self.assertAlmostEqual(y[0], self.y0, places=10)
        
        # 验证解的物理合理性（种群数量应该保持正值）
        self.assertTrue(np.all(x > 0))
        self.assertTrue(np.all(y > 0))
    
    def test_conservation_law_points_7(self):
        """测试守恒律 - 7分"""
        t, x, y = solve_lotka_volterra(self.alpha, self.beta, self.gamma, 
                                     self.delta, self.x0, self.y0, 
                                     self.t_span, self.dt)
        
        # Lotka-Volterra系统的守恒量：H = γx + βy - δln(x) - αln(y)
        H = (self.gamma * x + self.beta * y - 
             self.delta * np.log(x) - self.alpha * np.log(y))
        
        # 守恒量应该在数值误差范围内保持常数
        H_variation = np.max(H) - np.min(H)
        self.assertLess(H_variation, 0.01)  # 允许小的数值误差
    
    def test_periodic_behavior_points_5(self):
        """测试周期性行为 - 5分"""
        # 使用更长的时间来观察周期性
        t_span_long = (0, 20)
        t, x, y = solve_lotka_volterra(self.alpha, self.beta, self.gamma, 
                                     self.delta, self.x0, self.y0, 
                                     t_span_long, self.dt)
        
        # 检查是否回到接近初始状态（周期性的证据）
        # 寻找后半段时间内接近初始值的点
        mid_point = len(t) // 2
        x_later = x[mid_point:]
        y_later = y[mid_point:]
        
        # 找到接近初始值的点
        close_to_initial = np.where(
            (np.abs(x_later - self.x0) < 0.1) & 
            (np.abs(y_later - self.y0) < 0.1)
        )[0]
        
        # 应该存在接近初始状态的点（周期性证据）
        self.assertGreater(len(close_to_initial), 0)


class TestMethodComparison(unittest.TestCase):
    """测试方法比较功能"""
    
    def setUp(self):
        """设置测试参数"""
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 0.5
        self.delta = 2.0
        self.x0 = 2.0
        self.y0 = 2.0
        self.t_span = (0, 2)
        self.dt = 0.1
    
    def test_compare_methods_structure_points_5(self):
        """测试方法比较的数据结构 - 5分"""
        results = compare_methods(self.alpha, self.beta, self.gamma, 
                                self.delta, self.x0, self.y0, 
                                self.t_span, self.dt)
        
        # 验证返回结构
        self.assertIsInstance(results, dict)
        required_methods = ['euler', 'improved_euler', 'rk4']
        
        for method in required_methods:
            self.assertIn(method, results)
            self.assertIn('t', results[method])
            self.assertIn('x', results[method])
            self.assertIn('y', results[method])
            
            # 验证数组长度一致
            t_len = len(results[method]['t'])
            self.assertEqual(len(results[method]['x']), t_len)
            self.assertEqual(len(results[method]['y']), t_len)
    
    def test_method_accuracy_comparison_points_8(self):
        """测试方法精度比较 - 8分"""
        results = compare_methods(self.alpha, self.beta, self.gamma, 
                                self.delta, self.x0, self.y0, 
                                self.t_span, self.dt)
        
        # 所有方法应该有相同的初始条件
        for method in ['euler', 'improved_euler', 'rk4']:
            self.assertAlmostEqual(results[method]['x'][0], self.x0, places=10)
            self.assertAlmostEqual(results[method]['y'][0], self.y0, places=10)
        
        # 计算守恒量的变化（作为精度指标）
        def calculate_conservation_variation(x, y):
            H = (self.gamma * x + self.beta * y - 
                 self.delta * np.log(x) - self.alpha * np.log(y))
            return np.max(H) - np.min(H)
        
        var_euler = calculate_conservation_variation(
            results['euler']['x'], results['euler']['y'])
        var_ie = calculate_conservation_variation(
            results['improved_euler']['x'], results['improved_euler']['y'])
        var_rk4 = calculate_conservation_variation(
            results['rk4']['x'], results['rk4']['y'])
        
        # RK4应该最精确，欧拉法最不精确
        self.assertLess(var_rk4, var_ie)
        self.assertLess(var_ie, var_euler)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_small_time_step_points_3(self):
        """测试小时间步长 - 3分"""
        alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
        x0, y0 = 2.0, 2.0
        t_span = (0, 1)
        dt = 0.001  # 很小的步长
        
        t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, 
                                     x0, y0, t_span, dt)
        
        # 应该能正常运行且保持物理合理性
        self.assertTrue(np.all(x > 0))
        self.assertTrue(np.all(y > 0))
        self.assertAlmostEqual(x[0], x0, places=10)
        self.assertAlmostEqual(y[0], y0, places=10)
    
    def test_different_initial_conditions_points_4(self):
        """测试不同初始条件 - 4分"""
        alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
        t_span = (0, 5)
        dt = 0.01
        
        initial_conditions = [(1.0, 1.0), (0.5, 3.0), (4.0, 0.5)]
        
        for x0, y0 in initial_conditions:
            with self.subTest(x0=x0, y0=y0):
                t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, 
                                             x0, y0, t_span, dt)
                
                # 验证初始条件
                self.assertAlmostEqual(x[0], x0, places=10)
                self.assertAlmostEqual(y[0], y0, places=10)
                
                # 验证物理合理性
                self.assertTrue(np.all(x > 0))
                self.assertTrue(np.all(y > 0))
    
    def test_parameter_sensitivity_points_3(self):
        """测试参数敏感性 - 3分"""
        x0, y0 = 2.0, 2.0
        t_span = (0, 5)
        dt = 0.01
        
        # 测试不同参数组合
        parameter_sets = [
            (0.5, 0.25, 0.25, 1.0),  # 较小参数
            (2.0, 1.0, 1.0, 4.0),    # 较大参数
        ]
        
        for alpha, beta, gamma, delta in parameter_sets:
            with self.subTest(params=(alpha, beta, gamma, delta)):
                t, x, y = solve_lotka_volterra(alpha, beta, gamma, delta, 
                                             x0, y0, t_span, dt)
                
                # 基本合理性检查
                self.assertTrue(np.all(x > 0))
                self.assertTrue(np.all(y > 0))
                self.assertEqual(len(t), len(x))
                self.assertEqual(len(t), len(y))


def run_tests():
    """运行所有测试并返回结果"""
    # 创建测试套件
    test_classes = [
        TestLotkaVolterraSystem,
        TestNumericalMethods,
        TestLotkaVolterraSolver,
        TestMethodComparison,
        TestEdgeCases
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 计算总分
    total_score = 0
    max_score = 100
    
    if result.wasSuccessful():
        total_score = max_score
    else:
        # 根据通过的测试计算分数
        total_tests = result.testsRun
        failed_tests = len(result.failures) + len(result.errors)
        passed_tests = total_tests - failed_tests
        total_score = int((passed_tests / total_tests) * max_score)
    
    print(f"\n=== 测试结果汇总 ===")
    print(f"总测试数: {result.testsRun}")
    print(f"通过测试: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败测试: {len(result.failures)}")
    print(f"错误测试: {len(result.errors)}")
    print(f"总分: {total_score}/{max_score}")
    
    return result, total_score


if __name__ == '__main__':
    run_tests()