import numpy as np
import scipy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

gamma = 1.4
R = 287

N = 100  # grid
CFL = 0.2  # CFL number
x = np.linspace(0, 1, N)  # grid
dx = (x[-1]-x[0])/(N-1)  # grid distance

x_dis = 0.3  # discontinuity surface location
n_dis = x_dis*N

# w=[den, vel, press]
w_l = [1, 0.75, 1]
w_r = [0.125, 0, 0.1]


def u2w(u):
    w = u[:, 0]
    t = u[:, 1]
    m = u[:, 2]
    rho = w
    u = t / w
    p = (gamma - 1) * (m - 0.5*rho*u*u)
    out = np.hstack((rho.reshape((-1, 1)), u.reshape((-1, 1)), p.reshape((-1, 1))))
    return out


def w2u(w):
    rho = w[:, 0]
    u = w[:, 1]
    p = w[:, 2]
    rho_u = rho*u
    rho_e = p/(gamma-1)+0.5*rho*u*u
    out = np.hstack((rho.reshape((-1, 1)), rho_u.reshape((-1, 1)), rho_e.reshape((-1, 1))))
    return out


def w2f(w):
    rho = w[:, 0]
    u = w[:, 1]
    p = w[:, 2]
    f1 = rho * u
    f2 = rho * u * u + p
    f3 = p * u * gamma/(gamma-1)+0.5 * rho * u * u * u
    out = np.hstack((f1.reshape((-1, 1)), f2.reshape((-1, 1)), f3.reshape((-1, 1))))
    return out


def f_star(p_star, p, rho):
    # f_star function of Godunov Scheme
    # p_star: static pressure after the wave
    # p: static pressure before the wave
    # rho: density before the wave
    # f: value of f(p_star)
    t = p / R / rho  # ideal gas
    c = np.sqrt(gamma * R * t)
    if p_star > p:
        f = (p_star - p) / rho / c / np.sqrt((gamma + 1) / 2 / gamma * p_star / p + (gamma - 1) / 2 / gamma)
    else:
        f = 2 * c / (gamma - 1) * ((p_star / p) ** ((gamma - 1) / 2 / gamma) - 1)
    return f


def f_x(app, pp):
    def func(xin):
        return f_star(xin, pp[0], pp[1]) + f_star(xin, pp[2], pp[3]) - pp[4]
    res = fsolve(func, app)
    return res[0]


def godunov_scheme(u_l, u_r, t):
    # Computing F_i + 1 / 2 Using Godunov Scheme
    # u_l: [rho rho * u rho * E] on the left side
    # u_r: [rho rho * u rho * E] on the right side
    # t: time
    # F: [rho * u u ^ 2 + p rho * u * H]

    w__l = u2w(u_l)
    w__r = u2w(u_r)

    rho_1 = w__l[:, 0]
    u_1 = w__l[:, 1]
    p_1 = w__l[:, 2]
    rho_2 = w__r[:, 0]
    u_2 = w__r[:, 1]
    p_2 = w__r[:, 2]

    f_0 = f_star(0, p_1, rho_1) + f_star(0, p_2, rho_2)
    f_p1 = f_star(p_1, p_1, rho_1) + f_star(p_1, p_2, rho_2)
    f_p2 = f_star(p_2, p_1, rho_1) + f_star(p_2, p_2, rho_2)
    du = u_1 - u_2

    if du <= f_0:
        condition = 5
    elif f_0 < du <= min(f_p1, f_p2):
        condition = 4
    elif du > max(f_p1, f_p2):
        condition = 1
    else:
        if p_1 > p_2:
            condition = 2
        else:
            condition = 3

    pp = [p_1, rho_1, p_2, rho_2, du]
    p_star = f_x(0.5*(p_1+p_2), pp)
    u_star = 0.5 * (u_1 + u_2 + f_star(p_star, p_2, rho_2) - f_star(p_star, p_1, rho_1))

    t_1 = p_1 / R / rho_1  # ideal gas
    c_1 = np.sqrt(gamma * R * t_1)
    a_1 = rho_1 * c_1 * np.sqrt((gamma + 1) / 2 / gamma * p_star / p_1 + (gamma - 1) / 2 / gamma)
    t_2 = p_2 / R / rho_2  # ideal gas
    c_2 = np.sqrt(gamma * R * t_2)
    a_2 = rho_2 * c_2 * np.sqrt((gamma + 1) / 2 / gamma * p_star / p_2 + (gamma - 1) / 2 / gamma)

    if condition == 1 or condition == 3:
        rho_1_star = rho_1*a_1/(a_1-rho_1*(u_1-u_star))
        z_1h = u_1-a_1/rho_1
        z_1t = z_1h
    elif condition == 2 or condition == 4:
        c_1_star = c_1+(gamma-1)*(u_1-u_star)/2
        rho_1_star = gamma*p_star/(c_1_star**2)
        z_1h = u_1-c_1
        z_1t = u_star-c_1_star
    else:
        c_1_star = c_1 + (gamma - 1) * (u_1 - u_star) / 2
        rho_1_star = gamma * p_star / (c_1_star ** 2)
        z_1h = u_1 - c_1
        z_1t = u_1 - 2 / (gamma - 1) / c_1

    if condition == 1 or condition == 2:
        rho_2_star = rho_2*a_2/(a_2-rho_2*(u_star-u_2))
        z_2h = u_2+a_2/rho_2
        z_2t = z_2h
    elif condition == 3 or condition == 4:
        c_2_star = c_2+(gamma-1)*(u_star-u_2)/2
        rho_2_star = gamma*p_star/(c_2_star**2)
        z_2h = u_2+c_2
        z_2t = u_star+c_2_star
    else:
        c_2_star = c_2+(gamma-1)*(u_star-u_2)/2
        rho_2_star = gamma*p_star/(c_2_star**2)
        z_2h = u_2+c_2
        z_2t = u_2+2/(gamma-1)/c_2

    x1 = 0

    if x1 < z_1h * t:
        u = u_1
        p = p_1
        rho = rho_1
    elif z_1h * t <= x1 < z_1t * t:
        c_i = (gamma - 1) / (gamma + 1) * (u_1 - x1 / t) + 2 / (gamma + 1) * c_1
        u = x1 / t + c_i
        p = p_1 * (c_i / c_1) ** (2 * gamma / (gamma - 1))
        rho = gamma * p / (c_i ** 2)
    elif z_1t * t <= x1 < z_2t * t:
        u = u_star
        p = p_star
        if x1 < u_star * t:
            rho = rho_1_star
        else:
            rho = rho_2_star
    elif z_2t * t <= x1 < z_2h * t:
        c_i = (gamma - 1) / (gamma + 1) * (x1 / t - u_2) + 2 / (gamma + 1) * c_2
        u = x1 / t - c_i
        p = p_2 * (c_i / c_2) ** (2 * gamma / (gamma - 1))
        rho = gamma * p / (c_i ** 2)
    else:
        u = u_2
        p = p_2
        rho = rho_2

    w = np.hstack((rho.reshape(-1, 1), u.reshape(-1, 1), p.reshape(-1, 1)))
    f = w2f(w)
    return f


W = np.zeros((N, 3))
for i in range(N):
    if i <= n_dis:
        W[i, :] = w_l
    else:
        W[i, :] = w_r

U = w2u(W)
F = np.zeros((N-1, 3))  # flux vector
steps = 0
flag = 1
current_time = 0
t_max = 0.4
maxSteps = 1e4

u_left = U[0:N-1, :]
u_right = U[1:N, :]

rho_plot = []
U_plot = []
P_plot = []
T_plot = []

rho_plot.append(W[:, 0])
U_plot.append(W[:, 1])
P_plot.append(W[:, 2])
T_plot.append(current_time)

while flag:
    steps = steps + 1
    dt = CFL * dx / max(abs(W[:, 1]) + np.sqrt(gamma * W[:, 2] / W[:, 0]))
    if steps > maxSteps:  # stop criteria
        print('WARNING: maxSteps reached!')
        steps = steps - 1
        break
    if current_time+dt >= t_max:  # stop criteria
        dt = t_max-current_time
        flag = 0
    current_time = current_time + dt

    for i in range(N-1):
        F[i, :] = godunov_scheme(np.array([u_left[i, :]]), np.array([u_right[i, :]]), dt)

    for i in range(1, N-1):
        U[i, :] = U[i, :]-dt / dx * (F[i, :] - F[i - 1, :])

    U[0, :] = U[1, :]
    U[N-1, :] = U[N-2, :]

    W = u2w(U)

    rho_plot.append(W[:, 0])
    U_plot.append(W[:, 1])
    P_plot.append(W[:, 2])
    T_plot.append(current_time)

rho_plot = np.array(rho_plot)
U_plot = np.array(U_plot)
P_plot = np.array(P_plot)
T_plot = np.array(T_plot)

# 定义画布
fig, ax = plt.subplots(3, 1)
line1, = ax[0].plot([], [])
line2, = ax[1].plot([], [])
line3, = ax[2].plot([], [])


xtext_ani = ax[0].text(0.5, 0.5, "", fontsize=12)
# 获取直线的数组
def line_space1(B):
    x_plot = x
    return x_plot, rho_plot[B, :]


def line_space2(B):
    x_plot = x
    return x_plot, U_plot[B, :]


def line_space3(B):
    x_plot = x
    return x_plot, P_plot[B, :]


def update1(B):
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1.5)
    ax[0].set_ylabel("density")
    x_plot, y = line_space1(B)
    line1.set_data(x_plot, y)
    xtext_ani.set_text("t="+str(T_plot[B]))

    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1.5)
    ax[1].set_ylabel("velocity")
    x_plot, y = line_space2(B)
    line2.set_data(x_plot, y)

    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(0, 1.5)
    ax[2].set_ylabel("pressure")
    x_plot, y = line_space3(B)
    line3.set_data(x_plot, y)
    return [line1, xtext_ani], line2, line3


ani = FuncAnimation(fig, update1, frames=np.arange(steps), interval=50)
plt.show()

ani.save('move2.gif', writer='Pillow', fps=10)
