import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solution.lorenz_equations_solution import (
from lorenz_equations_student import (
    lorenz_system, solve_lorenz_equations, 
    compare_initial_conditions
)

class TestLorenzSystem(unittest.TestCase):
    def setUp(self):
        self.sigma = 10.0
        self.r = 28.0
        self.b = 8/3
        self.state = np.array([1.0, 1.0, 1.0])

    def test_lorenz_system_shape(self):
        """测试洛伦兹方程组返回值的形状"""
        result = lorenz_system(self.state, self.sigma, self.r, self.b)
        self.assertEqual(result.shape, (3,))

    def test_lorenz_system_values(self):
        """测试洛伦兹方程组在特定点的值"""
        expected = np.array([0.0, 26.0, -5/3])  # 修改后的正确预期值
        result = lorenz_system(np.array([1.0, 1.0, 1.0]), self.sigma, self.r, self.b)
        np.testing.assert_array_almost_equal(result, expected)

class TestLorenzSolver(unittest.TestCase):
    def setUp(self):
        self.params = {
            'sigma': 10.0,
            'r': 28.0,
            'b': 8/3,
            'x0': 0.1,
            'y0': 0.1,
            'z0': 0.1,
            't_span': (0, 1),
            'dt': 0.01
        }

    def test_solve_lorenz_equations(self):
        """测试洛伦兹方程求解器"""
        t, y = solve_lorenz_equations(**self.params)
        
        # 检查返回值的形状
        self.assertEqual(len(t), y.shape[1])
        self.assertEqual(y.shape[0], 3)

        # 检查初始值
        np.testing.assert_array_almost_equal(y[:,0], 
            [self.params['x0'], self.params['y0'], self.params['z0']])

class TestInitialConditions(unittest.TestCase):
    def test_compare_initial_conditions(self):
        """测试初始条件比较函数"""
        ic1 = (0.1, 0.1, 0.1)
        ic2 = (0.10001, 0.1, 0.1)
        
        # 确保函数可以正常调用而不抛出异常
        try:
            compare_initial_conditions(ic1, ic2)
        except Exception as e:
            self.fail(f"compare_initial_conditions() raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()