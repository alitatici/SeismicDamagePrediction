import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# create a 3x3 matrix
matrix = np.array([[0.1, 0.2, 0.15],
                   [0.05, 0.3, 0.1],
                   [0.1, 0.15, 0.0]])

# create a meshgrid for the x and y coordinates
x, y = np.meshgrid(np.arange(3), np.arange(3))

# flatten the matrix into a 1D array
z = matrix.flatten()

# create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 30)  # set the viewpoint
ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), 0.9, 0.9, z, color='b', alpha=1)

# add labels and ticks
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability')
ax.set_xticks(np.arange(3) + 0.5)
ax.set_yticks(np.arange(3) + 0.5)
ax.set_xticklabels(['A', 'B', 'C'])
ax.set_yticklabels(['X', 'Y', 'Z'])

# set tick labels to be centered
ax.tick_params(axis='both', which='major', labelsize=14)
plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), ha="center", rotation_mode="anchor")

# add title
ax.set_title("Joint Probability", fontsize=16)

# show plot
plt.show()
