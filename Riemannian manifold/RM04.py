import numpy as np
from sympy import Matrix, diag, symbols, lambdify

def generate_G(x, y):
    # 计算度量张量和克氏符号
    g = diag(1, 1) * Matrix([[x, y], [y, 1 - x]])
    G = g.inv()
    return lambdify((x, y), G, 'numpy')

# 生成符号变量
x, y = symbols('x y')

# 生成随机对称正定矩阵
M = Matrix([[x, y], [y, 1 - x]])
while not M.is_positive_definite:
    x_val = np.random.uniform(-1, 1)
    y_val = np.random.uniform(-1, 1)
    M = Matrix([[x_val, y_val], [y_val, 1 - x_val]])

# 显示生成的矩阵
print("生成的矩阵为：")
print(M)

# 计算度量张量和克氏符号
G_func = generate_G(x, y)
C = np.zeros((2, 2, 2))
for i in range(2):
    for j in range(i+1):
        for k in range(2):
            x_val = M[i, 0]
            y_val = M[i, 1]
            G_arr = G_func(x_val, y_val)
            C[i][j][k] = (1/2) * (G_arr[i,k].diff(symbols[j]) + G_arr[j,k].diff(symbols[i]) - G_arr[i,j].diff(symbols[k]))

# 显示计算结果
print("度量张量为：")
print(diag(1, 1) * M)
print("克氏符号为：")
print(C)
