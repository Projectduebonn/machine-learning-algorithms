import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

# 生成随机矩阵
M = np.random.rand(2,2)

# 转换成SymPy的符号类型
M_sym = sp.Matrix(M)

# 计算度量张量
G = M_sym.T * M_sym

# 计算逆度量张量
G_inv = G.inv()

# 计算Christoffel符号
symbols = sp.symbols('x y')
C = sp.zeros((2,2,2))
for i in range(2):
    for j in range(2):
        for k in range(2):
            C[i,j,k] = (1/2) * (G[i,k].diff(symbols[j]) + G[j,k].diff(symbols[i]) - G[i,j].diff(symbols[k]))

# 生成流形上的点
n = 100
x = np.linspace(-1,1,n)
y = np.linspace(-1,1,n)
X,Y = np.meshgrid(x,y)
points = np.array([X.flatten(), Y.flatten()]).T
Z = np.zeros(n**2)

# 计算测地线方程
def geodesic_equation(x, y, vx, vy):
    v = sp.Matrix([vx, vy])
    G_inv_arr = G_inv.subs({symbols[0]: x, symbols[1]: y})
    C_arr = C.subs({symbols[0]: x, symbols[1]: y})
    dvdt = - G_inv_arr * C_arr * v
    return [vx, vy, dvdt[0], dvdt[1]]

# 数值解测地线方程
t = np.linspace(0, 5, 1000)
sol = sp.solvers.ode.odeint(geodesic_equation, [0, 0, 1, 1], t)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol[:,0], sol[:,1], Z, color='r', lw=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()