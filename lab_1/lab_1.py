import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp

# Определение переменной и функций
t = sp.symbols('t')
r = 2 + sp.cos(6 * t)
phi = 7 * t + 1.2 * sp.cos(6 * t)

# Преобразование функций в численные
r_func = sp.lambdify(t, r, 'numpy')
phi_func = sp.lambdify(t, phi, 'numpy')

# Создание временного массива
t_vals = np.linspace(0, 2 * np.pi, 1000)

# Вычисление значений функций
r_vals = r_func(t_vals)
phi_vals = phi_func(t_vals)

# Преобразование в декартовы координаты
x_vals = r_vals * np.cos(phi_vals)
y_vals = r_vals * np.sin(phi_vals)

# Вычисление производных
r_dot = sp.diff(r, t)
phi_dot = sp.diff(phi, t)
r_ddot = sp.diff(r_dot, t)
phi_ddot = sp.diff(phi_dot, t)

# Преобразование производных в численные функции
r_dot_func = sp.lambdify(t, r_dot, 'numpy')
phi_dot_func = sp.lambdify(t, phi_dot, 'numpy')
r_ddot_func = sp.lambdify(t, r_ddot, 'numpy')
phi_ddot_func = sp.lambdify(t, phi_ddot, 'numpy')

# Вычисление значений производных
r_dot_vals = r_dot_func(t_vals)
phi_dot_vals = phi_dot_func(t_vals)
r_ddot_vals = r_ddot_func(t_vals)
phi_ddot_vals = phi_ddot_func(t_vals)

# Вычисление компонент скорости и ускорения
vx_vals = r_dot_vals * np.cos(phi_vals) - r_vals * np.sin(phi_vals) * phi_dot_vals
vy_vals = r_dot_vals * np.sin(phi_vals) + r_vals * np.cos(phi_vals) * phi_dot_vals
ax_vals = (r_ddot_vals - r_vals * phi_dot_vals**2) * np.cos(phi_vals) - (2 * r_dot_vals * phi_dot_vals + r_vals * phi_ddot_vals) * np.sin(phi_vals)
ay_vals = (r_ddot_vals - r_vals * phi_dot_vals**2) * np.sin(phi_vals) + (2 * r_dot_vals * phi_dot_vals + r_vals * phi_ddot_vals) * np.cos(phi_vals)

# Создание фигуры и осей
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
line, = ax.plot(x_vals, y_vals, lw=2)
arrow, = ax.plot([], [], 'r', marker='o')
radius_vector = ax.quiver(0, 0, 0, 0, color='r', scale=1, scale_units='xy', angles='xy')
velocity_arrow = ax.quiver(0, 0, 0, 0, color='g', scale=1, scale_units='xy', angles='xy')
acceleration_arrow = ax.quiver(0, 0, 0, 0, color='b', scale=1, scale_units='xy', angles='xy')

# Инициализация функции анимации
def init():
    arrow.set_data([], [])
    radius_vector.set_UVC(0, 0)
    velocity_arrow.set_UVC(0, 0)
    acceleration_arrow.set_UVC(0, 0)
    return arrow, radius_vector, velocity_arrow, acceleration_arrow

# Функция анимации
def animate(i):
    arrow.set_data(x_vals[i], y_vals[i])
    radius_vector.set_offsets([0, 0])
    radius_vector.set_UVC([x_vals[i]], [y_vals[i]])
    velocity_arrow.set_offsets([x_vals[i], y_vals[i]])
    velocity_arrow.set_UVC([vx_vals[i]], [vy_vals[i]])
    acceleration_arrow.set_offsets([x_vals[i], y_vals[i]])
    acceleration_arrow.set_UVC([ax_vals[i]], [ay_vals[i]])
    return arrow, radius_vector, velocity_arrow, acceleration_arrow

# Создание анимации
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t_vals), interval=20, blit=True)

# Отображение анимации
plt.show()