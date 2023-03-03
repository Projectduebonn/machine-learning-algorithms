from sympy import *
from sympy.diffgeom import Manifold, Patch, CoordSystem
from sympy.diffgeom import metric_to_Christoffel_1st, metric_to_Riemann_components

# 定义流形
manifold = Manifold("M", 2)
patch = Patch("P", manifold)

# 定义坐标系
cartesian = CoordSystem("cartesian", patch)

# 定义度量
x, y = symbols("x y")
a, b = symbols("a b", positive=True)
g = diag(1/a**2, 1/b**2)

# 应用度量于坐标系
cartesian.metric = g

# 计算克里斯托夫尔符号
ch = metric_to_Christoffel_1st(cartesian.metric)

# 计算黎曼曲率
R = metric_to_Riemann_components(cartesian.metric)

# 打印克里斯托夫尔符号和黎曼曲率
print("Christoffel symbols:")
for i, j, k in cartesian.cartesian_frame():
    print("C^{}_{}{} = {}".format(k, i, j, ch[k, i, j]))

print("\nRiemann curvature tensor:")
for i, j, k, l in cartesian.cartesian_frame():
    print("R^{}{}{}{} = {}".format(i, j, k, l, R[i, j, k, l]))
    