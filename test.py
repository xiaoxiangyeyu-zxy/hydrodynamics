import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Ntime = 100

a = []
for i in range(Ntime):
    a.append(np.linspace(0+0.1*i, 10+0.1*i, 100))
a = np.array(a)
# print(np.array(a))

# 定义画布
fig, ax = plt.subplots(2, 1)
line1, = ax[0].plot([], [])  # 返回的第一个值是update函数需要改变的
line2, = ax[1].plot([], [])  # 返回的第一个值是update函数需要改变的

xtext_ani = ax[0].text(0.5, 0.5, "", fontsize = 12)
# 获取直线的数组
def line_space1(B):
    x = np.linspace(0, 10, 100)
    return x, a[int(B), :]


def line_space2(B):
    x = np.linspace(0, 10, 100)
    return x, a[int(B), :]+5


# 这里B就是frame
def update1(B):
    ax[0].set_xlim(0, 10)
    ax[0].set_ylim(0, 20)
    x, y = line_space1(B)
    line1.set_data(x, y)
    xtext_ani.set_text("t="+str(B))

    ax[1].set_xlim(0, 10)
    ax[1].set_ylim(0, 30)
    x, y = line_space2(B)
    line2.set_data(x, y)

    return [line1, xtext_ani], line2


# 使用函数并保存保存会在下一篇文章讲
# 可以用plt.show()来代替一下

ani = FuncAnimation(fig, update1, frames=np.arange(Ntime), interval=50)
plt.show()

ani.save('move2.gif', writer='Pillow', fps=10)
