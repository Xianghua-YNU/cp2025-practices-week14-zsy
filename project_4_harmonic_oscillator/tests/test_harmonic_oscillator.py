import numpy as np
import unittest
from scipy.integrate import solve_ivp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solution.harmonic_oscillator_solution import (
from harmonic_oscillator_student import (
    harmonic_oscillator_ode,
    anharmonic_oscillator_ode,
    rk4_step,
    solve_ode,
    analyze_period
)

class TestHarmonicOscillator(unittest.TestCase):
    def setUp(self):
        """设置测试参数"""
        self.omega = 1.0
        self.dt = 0.01
        self.t_span = (0, 10)
        self.initial_state = np.array([1.0, 0.0])

    def test_harmonic_oscillator_ode(self):
        """测试简谐振子ODE函数"""
        state = np.array([1.0, 2.0])
        t = 0.0
        derivative = harmonic_oscillator_ode(state, t, self.omega)
        
        # 验证导数的形状
        self.assertEqual(derivative.shape, (2,))
        
        # 验证导数的值
        expected_dx = 2.0  # v
        expected_dv = -1.0  # -omega^2 * x
        np.testing.assert_almost_equal(derivative[0], expected_dx)
        np.testing.assert_almost_equal(derivative[1], expected_dv)

    def test_anharmonic_oscillator_ode(self):
        """测试非谐振子ODE函数"""
        state = np.array([1.0, 2.0])
        t = 0.0
        derivative = anharmonic_oscillator_ode(state, t, self.omega)
        
        # 验证导数的形状
        self.assertEqual(derivative.shape, (2,))
        
        # 验证导数的值
        expected_dx = 2.0  # v
        expected_dv = -1.0  # -omega^2 * x^3
        np.testing.assert_almost_equal(derivative[0], expected_dx)
        np.testing.assert_almost_equal(derivative[1], expected_dv)

    def test_rk4_step_accuracy(self):
        """测试RK4方法的精度"""
        # 使用简单的线性ODE进行测试
        def test_ode(y, t):
            return np.array([y[0]])

        initial_state = np.array([1.0])
        t = 0.0
        dt = 0.1

        # 计算RK4结果
        rk4_result = rk4_step(test_ode, initial_state, t, dt)

        # 计算解析解
        exact_solution = initial_state * np.exp(dt)

        # 验证RK4方法的精度（应该是4阶精度）
        error = np.abs(rk4_result - exact_solution)
        self.assertLess(error[0], 1e-5)

    def test_solve_ode_harmonic(self):
        """测试简谐振子的数值解"""
        t, states = solve_ode(
            harmonic_oscillator_ode,
            self.initial_state,
            self.t_span,
            self.dt,
            omega=self.omega
        )

        # 验证解的基本特性
        self.assertEqual(len(t), len(states))
        self.assertEqual(states.shape[1], 2)

        # 验证能量守恒（简谐振子的特性）
        # E = (1/2)mv^2 + (1/2)kx^2，这里m=1，k=omega^2
        initial_energy = 0.5 * self.initial_state[1]**2 + \
                        0.5 * self.omega**2 * self.initial_state[0]**2
        for state in states:
            energy = 0.5 * state[1]**2 + 0.5 * self.omega**2 * state[0]**2
            np.testing.assert_almost_equal(energy, initial_energy, decimal=3)

    def test_period_analysis(self):
        """测试周期分析函数"""
        # 生成测试数据：理想的简谐振动
        t = np.linspace(0, 20, 2000)
        x = np.cos(self.omega * t)
        v = -self.omega * np.sin(self.omega * t)
        states = np.column_stack((x, v))

        # 计算周期
        period = analyze_period(t, states)

        # 验证周期是否接近理论值 T = 2π/ω
        expected_period = 2 * np.pi / self.omega
        np.testing.assert_almost_equal(period, expected_period, decimal=2)

    def test_energy_conservation_anharmonic(self):
        """测试非谐振子的能量守恒"""
        t, states = solve_ode(
            anharmonic_oscillator_ode,
            self.initial_state,
            self.t_span,
            self.dt,
            omega=self.omega
        )

        # 计算初始能量 E = (1/2)mv^2 + (1/4)kx^4
        initial_energy = 0.5 * self.initial_state[1]**2 + \
                        0.25 * self.omega**2 * self.initial_state[0]**4

        # 验证能量守恒
        for state in states:
            energy = 0.5 * state[1]**2 + 0.25 * self.omega**2 * state[0]**4
            np.testing.assert_almost_equal(energy, initial_energy, decimal=3)

if __name__ == '__main__':
    unittest.main()