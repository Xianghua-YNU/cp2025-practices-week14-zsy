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
    theta, omega = state
    dtheta_dt = omega
    domega_dt = - (g / l) * np.sin(theta) + C * np.cos(theta) * np.sin(Omega * t)
    return [dtheta_dt, domega_dt]

def solve_pendulum(l=0.1, g=9.81, C=2, Omega=5, t_span=(0,100), y0=[0,0]):
    """
    求解受迫单摆运动方程
    返回: t, theta
    """
    # TODO: 使用solve_ivp求解受迫单摆方程
    # 提示: 需要调用forced_pendulum_ode函数
    sol = solve_ivp(
        fun=forced_pendulum_ode, 
        t_span=t_span, 
        y0=y0, 
        args=(l, g, C, Omega),
        dense_output=True,
        t_eval=t_eval
    )
    return sol.t, sol.y[0]

def find_resonance(l=0.1, g=9.81, C=2, Omega_range=None, t_span=(0,200), y0=[0,0]):
    """
    寻找共振频率
    返回: Omega_range, amplitudes
    """
    # TODO: 实现共振频率查找功能
    # 提示: 需要调用solve_pendulum函数并分析结果
    if Omega_range is None:
        Omega_0 = np.sqrt(g / l)
        Omega_range = np.linspace(Omega_0 / 2, 2 * Omega_0, 100)
    
    amplitudes = []
    
    for Omega in Omega_range:
        # 指定时间点确保有足够的数据点
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        t, theta = solve_pendulum(l, g, C, Omega, t_span, y0, t_eval)
        
        # 取稳态阶段的最大振幅
        if len(t) > 0:
            steady_state_index = t > 50
            if np.any(steady_state_index):
                max_amplitude = np.max(np.abs(theta[steady_state_index]))
            else:
                max_amplitude = 0
        else:
            max_amplitude = 0
        
        amplitudes.append(max_amplitude)
    
    return Omega_range, amplitudes
    
def plot_results(t, theta, title):
    """绘制结果"""
    # 此函数已提供完整实现，学生不需要修改
    plt.figure(figsize=(10, 5))
    plt.plot(t, theta)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    if savefig:
        plt.savefig(savefig)
    plt.show()

def main():
    """主函数"""
    # 任务1: 特定参数下的数值解与可视化
    # TODO: 调用solve_pendulum和plot_results
    l = 0.1
    g = 9.81
    C = 2
    Omega = 5
    t_span = (0, 100)
    y0 = [0, 0]
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    
    t, theta = solve_pendulum(l, g, C, Omega, t_span, y0, t_eval)
    plot_results(t, theta, title='θ(t) vs t for Ω=5, s^-1', savefig='task1.png')
    
    # 任务2: 探究共振现象
    # TODO: 调用find_resonance并绘制共振曲线
    Omega_range, amplitudes = find_resonance(l, g, C, Omega_range=None)
    
    plt.figure(figsize=(10, 5))
    plt.plot(Omega_range, amplitudes)
    plt.title('Amplitude vs Driving Frequency')
    plt.xlabel('Driving Frequency (rad/s)')
    plt.ylabel('Amplitude (rad)')
    plt.grid(True)
    plt.savefig('resonance.png')
    plt.show()
    
    # 找到共振频率并绘制共振情况
    # TODO: 实现共振频率查找和绘图
    if amplitudes:
        resonance_index = np.argmax(amplitudes)
        Omega_res = Omega_range[resonance_index]
        
        t, theta = solve_pendulum(l, g, C, Omega_res, t_span, y0, t_eval)
        plot_results(t, theta, title=f'θ(t) vs t at resonance frequency Ω={Omega_res:.2f} s^-1', savefig='resonance_case.png')

if __name__ == '__main__':
    main()
