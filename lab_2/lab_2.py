import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры системы
mass = 1.0                 # масса материальных точек
rod_length1 = 1.0          # длина первого стержня
rod_length2 = 1.0          # длина второго стержня
spring_constant = 0.1      # жёсткость пружины
gravity = 9.81             # ускорение свободного падения
damping_coefficient = 0.1  # Дампинг

# Начальные условия
initial_phi = np.pi / 6    # Начальное отклонение первого маятника
initial_psi = np.pi / 2    # Начальное отклонение второго маятника
initial_omega1 = 0.0       # Начальная угловая скорость первого маятника
initial_omega2 = 0.0       # Начальная угловая скорость второго маятника

# Временной интервал
dt = 0.005                 # Шаг интегрирования
time_span = (0, 20.0)
time_eval = np.arange(time_span[0], time_span[1], dt)

# Уравнения движения
def equations(t, y):
    """
    y = [phi, psi, omega1, omega2]
    phi, psi: углы отклонения
    omega1, omega2: угловые скорости
    """
    phi, psi, omega1, omega2 = y

    # Координаты точек M1 и M2
    x1 = rod_length1 * np.sin(phi)
    y1 = -rod_length1 * np.cos(phi)
    x2 = x1 + rod_length2 * np.sin(psi)
    y2 = y1 - rod_length2 * np.cos(psi)

    # Длина пружины
    spring_length = np.sqrt(x1**2 + y1**2)

    # Сила пружины и её проекции
    if spring_length == 0:
        fx_spring = 0
        fy_spring = 0
    else:
        spring_force = -spring_constant * (spring_length - rod_length1) / spring_length
        fx_spring = spring_force * x1
        fy_spring = spring_force * y1

    # Угловые ускорения
    domega1 = (
        -mass * gravity * rod_length1 * np.sin(phi)
        + fx_spring * rod_length1 * np.cos(phi)
        + fy_spring * rod_length1 * np.sin(phi)
        - damping_coefficient * omega1
        - spring_constant * omega1
    ) / (mass * rod_length1**2)

    domega2 = (
        -mass * gravity * rod_length2 * np.sin(psi)
        - damping_coefficient * omega2
        - spring_constant * omega2
    ) / (mass * rod_length2**2)

    return [omega1, omega2, domega1, domega2]

# Начальные условия
initial_conditions = [initial_phi, initial_psi, initial_omega1, initial_omega2]

# Численное решение

solution = solve_ivp(equations, time_span, initial_conditions, t_eval=time_eval, method='RK45')

# Анимация
fig, ax = plt.subplots(figsize=(10, 6))

# Создаем объекты для анимации
# "брусок"
block_width, block_height = 0.1, 0.2
block = plt.Rectangle((-block_width/2, -block_height/2), 
                      block_width, block_height, 
                      color='gray', zorder=5)
ax.add_patch(block)

# Рисуем "стенки"
left_wall = plt.plot([-1.5 - block_width / 2 - 0.03, -1.5 - block_width / 2 - 0.03], [-0.5, -1.2], color='k', linewidth=1)
right_wall = plt.plot([-1.5 + block_width / 2 + 0.03, -1.5 + block_width / 2 + 0.03], [-0.5, -1.2], color='k', linewidth=1)


rod1, = ax.plot([], [], 'o-', lw=2, markersize=10)
rod2, = ax.plot([], [], 'o-', lw=2, markersize=10)
spring_line, = ax.plot([], [], 'b-', lw=2)

ax.set_xlim(-rod_length1 - rod_length2 - 0.5, rod_length1 + rod_length2 + 0.5)
ax.set_ylim(-rod_length1 - rod_length2 - 0.5, rod_length1 + rod_length2 + 0.5)
ax.set_aspect('equal')
ax.grid(True)

# Функция для вычисления координат зигзага
def get_spring_zigzag(x_start, y_start, x_end, y_end, turns=8, amplitude=0.1):
    # Делаем разбиение по длине
    num_points = 2 * turns + 1
    xs = np.linspace(x_start, x_end, num_points)
    ys = np.linspace(y_start, y_end, num_points)
    # Смещаем точки по нормали для зигзага
    zigzag_x = []
    zigzag_y = []
    for i in range(num_points):
        # пропорция вдоль пружины
        t = i / (num_points - 1)
        # направление
        dx = x_end - x_start
        dy = y_end - y_start
        length = np.sqrt(dx**2 + dy**2)
        # нормаль
        nx = -dy / length
        ny = dx / length
        # смещение
        offset = amplitude if i % 2 == 1 else -amplitude
        # координаты
        zx = xs[i] + offset * nx
        zy = ys[i] + offset * ny
        zigzag_x.append(zx)
        zigzag_y.append(zy)
    return zigzag_x, zigzag_y

def init():
    rod1.set_data([], [])
    rod2.set_data([], [])
    spring_line.set_data([], [])
    block.set_xy((-block_width/2, -block_height/2))
    return rod1, rod2, spring_line, block

def update(frame):
    phi, psi, omega1, omega2 = solution.y[:, frame]
    
    x1 = rod_length1 * np.sin(phi)
    y1 = -rod_length1 * np.cos(phi)
    x2 = x1 + rod_length2 * np.sin(psi)
    y2 = y1 - rod_length2 * np.cos(psi)
    
    rod1.set_data([0, x1], [0, y1])
    rod2.set_data([x1, x2], [y1, y2])

    # Зигзаг от центра (там где брусок) до первой массы
    zx, zy = get_spring_zigzag(-1.5, y1, x1, y1, turns=6, amplitude=0.05)
    spring_line.set_data(zx, zy)

    # Брусок в конце пружины
    block.set_xy((-1.5 - block_width/2, y1 - block_height/2))
    
    return rod1, rod2, spring_line, block

ani = FuncAnimation(fig, update, frames=len(time_eval), init_func=init, blit=True, interval=5)

plt.show()
