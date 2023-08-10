import matplotlib.pyplot as plt
import numpy as np

vel = np.load('saved_vel.npy')[:, 0]
print(vel)
x = range(0, len(vel))


plt.plot(x, vel)
plt.show()

