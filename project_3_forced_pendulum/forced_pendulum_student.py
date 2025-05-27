import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def forced_pendulum_ode(t, state, l, g, C, Omega):
    """
    受驱单摆的常微分方程
    state: [theta, omega]
    返回: [dtheta/dt, domega/dt]
    """
    # TODO: 在此实现受迫单摆的ODE方程
    raise NotImplementedError("请实现此函数")

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0,100), y0=[0,0]):
    """
    求解受迫单摆运动方程
    返回: t, theta
    """
    # TODO: 使用solve_ivp求解受迫单摆方程
    # 提示: 需要调用forced_pendulum_ode函数
    raise NotImplementedError("请实现此函数")

def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_span=(0,200), y0=[0,0]):
    """
    寻找共振频率
    返回: Omega_range, amplitudes
    """
    # TODO: 实现共振频率查找功能
    # 提示: 需要调用solve_pendulum函数并分析结果
    raise NotImplementedError("请实现此函数")

def plot_results(t, theta, title):
    """绘制结果"""
    # 此函数已提供完整实现，学生不需要修改
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()

def main():
    """主函数"""
    # 任务1: 特定参数下的数值解与可视化
    # TODO: 调用solve_pendulum和plot_results
    
    # 任务2: 探究共振现象
    # TODO: 调用find_resonance并绘制共振曲线
    
    # 找到共振频率并绘制共振情况
    # TODO: 实现共振频率查找和绘图

if __name__ == '__main__':
    main()