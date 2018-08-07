
from kNN import *

import matplotlib
import matplotlib.pyplot as plt


dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0*array(dating_labels), 15.0*array(dating_labels))
plt.show()

