import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

# 输入去年12个月的销售量
sales_last_year = np.array([100, 120, 130, 140, 150, 170, 180, 190, 200, 210, 220, 230])

# 计算训练周期数
num_periods = len(sales_last_year) // 3

# 将销售量数据分为训练集和测试集
X_train = []
y_train = []
X_test = []
y_test = []
for i in range(num_periods - 1):
    X_train.append(sales_last_year[i*3:i*3+3])
    y_train.append(sales_last_year[i*3+3])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array([sales_last_year[-3:]])
y_test = np.array([sales_last_year[-2], sales_last_year[-1]])

# 建立线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测今年每个月的销售量
predictions = []
for i in range(num_periods):
    if i == num_periods - 1:
        X_test = np.array([sales_last_year[-3:]])
    else:
        X_test = np.array([sales_last_year[i*3:i*3+3]])
    y_pred = model.predict(X_test)
    predictions.append(y_pred[0])
    sales_last_year = np.append(sales_last_year, y_pred[0])

# 输出预测结果
print("预测结果：", predictions)

predictions = [math.ceil(p) for p in predictions]

# 画出直方图
x_ticks = range(1, len(predictions) + 1)
for i, v in enumerate(predictions):
    plt.text(x_ticks[i] - 0.1, v + 1, str(v))
plt.bar(x_ticks, predictions)
plt.title("Monthly Sales Predictions")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(x_ticks)
plt.show()