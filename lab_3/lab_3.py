import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Исходные данные системы
m = 0.1  # масса в кг
l1 = 0.5  # длина первого стержня в м
l2 = 0.5  # длина второго стержня в м
c = 5.0   # жёсткость пружины в Н/м
g = 9.81  # ускорение свободного падения в м/с^2

# Начальные условия
phi0 = np.pi / 10
psi0 = np.pi / 10
phi_dot0 = 0
psi_dot0 = 0

# Уравнения движения
def equations(t, y):
    phi, psi, phi_dot, psi_dot = y

    # Матрица коэффициентов
    A = np.array([
        [2 * l1, l2 * np.cos(psi - phi)],
        [l1 * np.cos(psi - phi), l2]
    ])

    # Правая часть
    b = np.array([
        -2 * g * np.sin(phi) - (c * l1 / m) * np.cos(phi) * np.sin(phi)
        - l2 * psi_dot**2 * np.sin(psi - phi),
        -g * np.sin(psi) + l1 * phi_dot**2 * np.sin(psi - phi)
    ])

    # Решение системы уравнений
    phi_ddot, psi_ddot = np.linalg.solve(A, b)

    return [phi_dot, psi_dot, phi_ddot, psi_ddot]

# Время моделирования
t_span = (0, 10)  # от 0 до 10 секунд
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Начальный вектор состояния
y0 = [phi0, psi0, phi_dot0, psi_dot0]

# Решение системы
solution = solve_ivp(equations, t_span, y0, t_eval=t_eval, method='RK45')

# Извлечение решения
phi, psi, phi_dot, psi_dot = solution.y

# Координаты масс для анимации
x1 = l1 * np.sin(phi)
y1 = -l1 * np.cos(phi)
x2 = x1 + l2 * np.sin(psi)
y2 = y1 - l2 * np.cos(psi)

# Вычисление нормальной силы N
phi_ddot = np.gradient(phi_dot, t_eval)
psi_ddot = np.gradient(psi_dot, t_eval)
N = m * (g * np.cos(psi) - l1 * (phi_ddot * np.sin(psi - phi) - phi_dot**2 * np.cos(psi - phi)) + l2 * psi_dot**2)

# Построение графиков
fig_graphs, axs = plt.subplots(3, 1, figsize=(8, 10))
axs[0].plot(t_eval, phi, label="phi(t)")
axs[0].set_title("График phi(t)")
axs[0].set_xlabel("Время (с)")
axs[0].set_ylabel("phi (рад)")
axs[0].grid()
axs[0].legend()

axs[1].plot(t_eval, psi, label="psi(t)", color='orange')
axs[1].set_title("График psi(t)")
axs[1].set_xlabel("Время (с)")
axs[1].set_ylabel("psi (рад)")
axs[1].grid()
axs[1].legend()

axs[2].plot(t_eval, N, label="N(t)", color='green')
axs[2].set_title("График N(t)")
axs[2].set_xlabel("Время (с)")
axs[2].set_ylabel("N (Н)")
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()

# Анимация
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
spring, = ax.plot([], [], '-', lw=1, color='red')

# Инициализация анимации
def init():
    line.set_data([], [])
    spring.set_data([], [])
    return line, spring

# Обновление кадров
def update(frame):
    this_x = [0, x1[frame], x2[frame]]
    this_y = [0, y1[frame], y2[frame]]
    spring_x = [-0.5, x1[frame]]
    spring_y = [y1[frame], y1[frame]]
    line.set_data(this_x, this_y)
    spring.set_data(spring_x, spring_y)
    return line, spring

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20)

plt.show()