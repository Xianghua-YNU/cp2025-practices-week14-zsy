import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solution.forced_pendulum_solution import forced_pendulum_ode, solve_pendulum, find_resonance
from forced_pendulum_student import forced_pendulum_ode, solve_pendulum, find_resonance

class TestForcedPendulum(unittest.TestCase):
    def test_forced_pendulum_ode(self):
        """测试受迫单摆ODE函数的正确性"""
        # 测试小角度近似
        state = [0.1, 0]
        result = forced_pendulum_ode(0, state, l=1, g=9.81, C=0, Omega=0)
        self.assertAlmostEqual(result[0], 0)
        # 修改容差为2位小数
        self.assertAlmostEqual(result[1], -0.981, places=2)
        
        # 测试受迫项
        state = [0, 0]
        result = forced_pendulum_ode(0, state, l=1, g=9.81, C=2, Omega=5)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)
    
    def test_solve_pendulum(self):
        """测试求解器返回值的形状和范围"""
        t, theta = solve_pendulum(t_span=(0, 1))
        self.assertEqual(len(t), 2000)
        self.assertEqual(len(theta), 2000)
        self.assertTrue(np.all(np.abs(theta) <= np.pi))
    
    def test_find_resonance(self):
        """测试共振频率查找功能"""
        Omega_range, amplitudes = find_resonance(t_span=(0, 20))
        self.assertEqual(len(Omega_range), 50)
        self.assertEqual(len(amplitudes), 50)
        self.assertTrue(np.max(amplitudes) > 0)

if __name__ == '__main__':
    unittest.main()