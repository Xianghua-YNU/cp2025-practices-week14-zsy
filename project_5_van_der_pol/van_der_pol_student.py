import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List
from scipy.integrate import solve_ivp

def van_der_pol_ode(state: np.ndarray, t: float, mu: float = 1.0, omega: float = 1.0) -> np.ndarray:
    """
    van der Pol振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        mu: float, 非线性阻尼参数
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    # TODO: 实现van der Pol方程
    # dx/dt = v
    # dv/dt = mu(1-x^2)v - omega^2*x
    dxdt = v
    dvdt = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dxdt, dvdt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    # TODO: 实现RK4方法
    k1 = ode_func(t, state, **kwargs)
    k2 = ode_func(t + 0.5 * dt, state + 0.5 * dt * k1, **kwargs)
    k3 = ode_func(t + 0.5 * dt, state + 0.5 * dt * k2, **kwargs)
    k4 = ode_func(t + dt, state + dt * k3, **kwargs)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    # TODO: 实现ODE求解器
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = solve_ivp(ode_func, t_span, initial_state, t_eval=t_eval, args=tuple(kwargs.values()), method='RK45')
    return sol.t, sol.y.T


def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现时间演化图的绘制
    plt.figure()
    plt.plot(t, states[:, 0], label='x(t)')
    plt.plot(t, states[:, 1], label='v(t)')
    plt.xlabel('t')
    plt.ylabel('State')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title}.png")  # 保存为PNG
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现相空间图的绘制
    plt.figure()
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('displacement x')
    plt.ylabel('velocity v')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{title}.png")  # 保存为PNG
    plt.show()

def calculate_energy(state: np.ndarray, omega: float = 1.0) -> float:
    """
    计算van der Pol振子的能量。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        omega: float, 角频率
    
    返回:
        float: 系统的能量
    """
    # TODO: 实现能量计算
    # E = (1/2)v^2 + (1/2)omega^2*x^2
    x, v = state
    E = 0.5 * v**2 + 0.5 * omega**2 * x**2
    return E

def analyze_limit_cycle(states: np.ndarray) -> Tuple[float, float]:
    """
    分析极限环的特征（振幅和周期）。
    
    参数:
        states: np.ndarray, 状态数组
        dt: float, 时间步长
    
    返回:
        Tuple[float, float]: (振幅, 周期)
    """
    # TODO: 实现极限环分析
    skip = int(len(states)*0.5)
    x = states[skip:, 0]
    t = np.arange(len(x))
    
    # 计算振幅（取最大值的平均）
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(x[i])
    amplitude = np.mean(peaks) if peaks else np.nan
    
    # 计算周期（取相邻峰值点的时间间隔平均）
    if len(peaks) >= 2:
        periods = np.diff(t[1:-1][np.array([x[i] > x[i-1] and x[i] > x[i+1] for i in range(1, len(x)-1)])])
        period = np.mean(periods) if len(periods) > 0 else np.nan
    else:
        period = np.nan
    
    return amplitude, period
    
def main():
    # 设置基本参数
    mu = 1.0
    omega = 1.0
    t_span = (0, 20)
    dt = 0.01
    initial_state = np.array([1.0, 0.0])
    
    # TODO: 任务1 - 基本实现
    # 1. 求解van der Pol方程
    # 2. 绘制时间演化图
    t_values, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    plot_time_evolution(t_values, states, title="Task 1 - Time Evolution Graph")
    # TODO: 任务2 - 参数影响分析
    # 1. 尝试不同的mu值
    # 2. 比较和分析结果
    mu_values = [1.0, 2.0, 4.0]
    for mu in mu_values:
        t_values, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_time_evolution(t_values, states, title=f"Task 2 - mu={mu} Time Evolution Graph")
        amplitude, period = analyze_limit_cycle(states, dt)
        print(f"mu={mu}: 振幅={amplitude:.2f}, 周期={period:.2f}")
        plot_phase_space(states, title=f"Task 2 - mu={mu} Phase Space Trajectory")

    # TODO: 任务3 - 相空间分析
    # 1. 绘制相空间轨迹
    # 2. 分析极限环特征
    for mu in mu_values:
        t_values, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
        plot_phase_space(states, title=f"Task 3 - mu={mu} Phase Space Trajectory")
        amplitude, period = analyze_limit_cycle(states, dt)
        print(f"mu={mu}: 振幅={amplitude:.2f}, 周期={period:.2f}")

    
    # TODO: 任务4 - 能量分析
    # 1. 计算和绘制能量随时间的变化
    # 2. 分析能量的耗散和补充
    t_values, states = solve_ode(van_der_pol_ode, initial_state, t_span, dt, mu=mu, omega=omega)
    energies = np.array([calculate_energy(state, omega) for state in states])
    plt.figure()
    plt.plot(t_values, energies)
    plt.xlabel('t')
    plt.ylabel('Energy E')
    plt.title("Task 4 - Changes in Energy over Time")
    plt.grid(True)
    plt.savefig("Task 4 - Energy Changes over Time.png")  # 保存为PNG
    plt.show()
 
    plt.figure()
    plt.plot(t_values[:-1], np.diff(energies) / dt)
    plt.xlabel('t')
    plt.ylabel('energy gradient dE/dt')
    plt.title("Task 4 - Rate of Energy Change")
    plt.grid(True)
    plt.savefig("Task 4 - Rate of Energy Change.png")  # 保存为PNG
    plt.show()

if __name__ == "__main__":
    main()
