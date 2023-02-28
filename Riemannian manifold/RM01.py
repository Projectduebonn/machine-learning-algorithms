import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义黎曼度量和坐标函数
def metric(x, y):
    return np.array([[1, 0], [0, 1]])

def coord_func(x, y):
    return np.array([x, y])

# 生成网格
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)

# 计算每个网格点的坐标
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = np.dot(coord_func(X[i,j], Y[i,j]), np.dot(metric(X[i,j], Y[i,j]), coord_func(X[i,j], Y[i,j])))

# 绘制图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()
