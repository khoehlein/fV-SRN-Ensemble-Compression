import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

data = np.random.randn(10000000)
print('Building tree')
tree = KDTree(data[:, None])

print('Searching neighbors')
k = 100
distances, _ = tree.query(data[:, None], k=[k], workers=9)
min_distance = distances.min()
max_distance = distances.max()
p = k / (2 * len(data) * distances)

print('Plotting')
plt.scatter(- data ** 2 / 2 - math.log(2 * np.pi) / 2., np.log(p), alpha=0.01)
plt.show()

print('Finished')




