import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

# 创建数据
x1 = [1, 2, 3]
y1 = [4, 5, 6]
x2 = [4, 5, 6]
y2 = [7, 8, 9]

# 绘制散点图
scatter1 = plt.scatter(x1, y1, color='red', label='Point Type A')
scatter2 = plt.scatter(x2, y2, color='blue', label='Point Type A')

# 创建自定义图例
legend_elements = [(scatter1, scatter2)]
labels = ['Point Type A']
plt.legend(handles=legend_elements, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None)})

# 显示图形
plt.show()