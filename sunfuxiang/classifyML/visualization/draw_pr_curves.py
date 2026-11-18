import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
plt.title('P-R Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
y_true = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 1])
y_scores = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# y_true为样本实际的类别1为正例0为反例，y_scores为阈值
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
plt.plot(recall, precision)
plt.savefig("./figure2/vitPRCurvesPaddy.pdf")
plt.show()