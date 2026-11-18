# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 读取存储为txt文件的数据
# def data_read(dir_path):
#     with open(dir_path, "r") as f:
#         raw_data = f.read()
#         data = raw_data[1:-1].split(", ")
#
#     return np.asfarray(data, float)
#
#
# # 不同长度数据，统一为一个标准，倍乘x轴
# def multiple_equal(x, y):
#     x_len = len(x)
#     y_len = len(y)
#     times = x_len/y_len
#     y_times = [i * times for i in y]
#     return y_times
#
#
# if __name__ == "__main__":
#
#     val_acc1_path = r"D:\curves\pvt_cabbage_loss.txt"
#     val_acc2_path = r"D:\curves\pvtet_cabbage_loss.txt"
#     # val_acc3_path = r"./result/pvt_flower17_val_acc.txt"
#     # val_acc4_path = r"./result/pvtet_flower17_val_acc.txt"
#     # val_acc5_path = r"./result/swin_flower17_val_acc.txt"
#     # val_acc6_path = r"./result/swinet_flower17_val_acc.txt"
#
#
#     y_val_acc1 = data_read(val_acc1_path)
#     y_val_acc2 = data_read(val_acc2_path)
#     # y_val_acc3 = data_read(val_acc3_path)
#     # y_val_acc4 = data_read(val_acc4_path)
#     # y_val_acc5 = data_read(val_acc5_path)
#     # y_val_acc6 = data_read(val_acc6_path)
#
#
#     # 去除顶部和右边框框
#     ax = plt.axes()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     x1 = range(1, 101)
#
#     plt.xlabel('Epochs', fontsize=15)
#     plt.ylabel('Validation loss', fontsize=15)
#     plt.plot(x1, y_val_acc1, color='blue', label="PVT")
#     plt.plot(x1, y_val_acc2, color='red', label="PVT+EFG")
#     # plt.plot(x1, y_val_acc3, color='green', label="PVT")
#     # plt.plot(x1, y_val_acc4, color='orange', label="PVT+efg")
#     # plt.plot(x1, y_val_acc5, color='blue', label="Swin")
#     # plt.plot(x1, y_val_acc6, color='red', label="Swin+efg")
#     plt.legend()
#     plt.grid()
#     plt.savefig("./figure2/pvtCurvesCabbage.pdf")
#     plt.show()
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimHei']

plt.rcParams['axes.unicode_minus']=False

# Read the text file
data = []
with open('D:\\aaa.txt', 'r') as file:
    for line in file:
        # Split the line by whitespace
        values = line.split()
        # Convert the values to float and append to the data list
        data.append([float(value) for value in values])

# Extract X and Y values
x = [row[0] for row in data]
y = [row[1:6] for row in data]

# Transpose the Y values
y = list(map(list, zip(*y)))

# Plot the line graph
plt.plot(x, y[0], label='K-Means', color='blue')
plt.plot(x, y[1], label='K-Means+', color='black')
plt.plot(x, y[2], label='GMM', color='orange')
plt.plot(x, y[3], label='本文算法', color='red')
plt.plot(x, y[4], label='Agglomerative Clustering', color='black')

# Add labels and legend
plt.title('服务器部署成本')
plt.xlabel('部署成本（元）')
plt.ylabel('聚类数量')
plt.legend()

# Display the plot
plt.savefig('D:\\cost.png')
plt.show()


