import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    # TODO: 实现简谐振子的微分方程组
    # dx/dt = v
    # dv/dt = -omega^2 * x
    x, v = state
    dxdt = v
    dvdt = -omega**2 * x
    return np.array([dxdt, dvdt])

def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    # TODO: 实现非谐振子的微分方程组
    # dx/dt = v
    # dv/dt = -omega^2 * x^3
    x, v = state
    dxdt = v
    dvdt = -omega2 * x3
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
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + dt/2 * k1, t + dt/2, **kwargs)
    k3 = ode_func(state + dt/2 * k2, t + dt/2, **kwargs)
    k4 = ode_func(state + dt * k3, t + dt, **kwargs)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

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
    t_start, t_end = t_span
    n_steps = int((t_end - t_start)/dt) + 1
    t = np.linspace(t_start, t_end, n_steps)
    states = np.zeros((n_steps, len(initial_state)))
    states[0] = initial_state
    current_state = initial_state.copy()
    for i in range(1, n_steps):
        current_state = rk4_step(ode_func, current_state, t[i-1], dt, **kwargs)
        states[i] = current_state

    return t, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现时间演化图的绘制
    plt.figure(figsize=(10, 6))
    plt.plot(t, states[:, 0], label='位移x(t)')
    plt.plot(t, states[:, 1], label='速度v(t)')
    plt.xlabel('时间 t')
    plt.ylabel('位移 x 和速度 v')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title}.png")  # 保存图像
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现相空间图的绘制
    plt.figure(figsize=(10, 6))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('位移 x')
    plt.ylabel('速度 v')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{title}.png")  # 保存图像
    plt.show()
    
def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
    
    返回:
        float: 估计的振动周期
    """
    # TODO: 实现周期分析
    x = states[:, 0]
    peaks, _ = find_peaks(x)
    if len(peaks) < 2:
        return np.nan  # 返回无效值表示无法计算周期
    
    time_diffs = np.diff(t[peaks])
    period = np.mean(time_diffs)
    return period

def main():
    # 设置参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    
    # TODO: 任务1 - 简谐振子的数值求解
    # 1. 设置初始条件 x(0)=1, v(0)=0
    # 2. 求解方程
    # 3. 绘制时间演化图
    initial_state_harmonic = np.array([1.0, 0.0])
    t_harmonic, states_harmonic = solve_ode(harmonic_oscillator_ode, initial_state_harmonic, t_span, dt, omega=omega)
    plot_time_evolution(t_harmonic, states_harmonic, "简谐振子的时间演化")
    plot_phase_space(states_harmonic, "简谐振子的相空间轨迹")
    
    # TODO: 任务2 - 振幅对周期的影响分析
    # 1. 使用不同的初始振幅
    # 2. 分析周期变化
    initial_amplitudes = [1.0, 2.0]
    periods_harmonic = []

    for amp in initial_amplitudes:
        initial_state = np.array([amp, 0.0])
        t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        periods_harmonic.append(period)

    print(f"简谐振子不同振幅下的周期: {periods_harmonic}")
    # TODO: 任务3 - 非谐振子的数值分析
    # 1. 求解非谐振子方程
    # 2. 分析不同振幅的影响
    initial_state_anharmonic = np.array([1.0, 0.0])
    t_anharmonic, states_anharmonic = solve_ode(anharmonic_oscillator_ode, initial_state_anharmonic, t_span, dt, omega=omega)
    plot_time_evolution(t_anharmonic, states_anharmonic, "非谐振子的时间演化")
    plot_phase_space(states_anharmonic, "非谐振子的相空间轨迹")

    initial_amplitudes_anharmonic = [1.0, 2.0]
    periods_anharmonic = []

    for amp in initial_amplitudes_anharmonic:
        initial_state = np.array([amp, 0.0])
        t, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        periods_anharmonic.append(period)

    print(f"非谐振子不同振幅下的周期: {periods_anharmonic}")
    # TODO: 任务4 - 相空间分析
    # 1. 绘制相空间轨迹
    # 2. 比较简谐和非谐振子
    plot_phase_space(states_harmonic, "简谐振子相空间轨迹")
    plot_phase_space(states_anharmonic, "非谐振子相空间轨迹")

if __name__ == "__main__":
    main()
