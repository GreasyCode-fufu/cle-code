# import numpy as np
# from matplotlib import pyplot as plt
#
# n = 7
# X = np.arange(n)
# width = 0.36
# Y1 = [0.97, 0.92, 0.66, 0.96, 0.98, 0.98, 1.00]
# ##317DB7  #26BED0   lightseagreen
# plt.bar(X - width / 2, Y1, width=width, facecolor='#26BED0', edgecolor='white', align='edge')
# plt.xlabel('Classes')
# plt.ylabel('Recall')
#
# for x, y in zip(X, Y1):
#     plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')
#
# plt.ylim(0, +1.1)
# plt.savefig('./figure2/fig8b.jpg')
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

size = 3
a = [0.95, 1.00, 0.93]
b = [0.91, 0.95, 0.91]
x = np.arange(size)
x_labels = ["XXS", "XS", "S"]

total_width, n = 0.8, 3
width = total_width / n

plt.xticks(x, x_labels)
plt.ylabel('Precision')

x = x - (total_width - width) / 3

for i, j in zip(x, a):
    plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
for i, j in zip(x + width, b):
    plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)

plt.bar(x, a, width=width, color='#26BED0', label='0')
plt.bar(x + width, b, width=width, color='#317DB7', label='1')

# Position the legend at the top and specify the horizontal layout
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

plt.savefig('./figure2/12a.jpg')
plt.show()

