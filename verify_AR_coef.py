# %% imports
import matplotlib.pyplot as plt
import numpy as np

from routine.simulation import AR2tau, tau2AR
from routine.utilities import norm

# %% verify AR process
tau_d = 6
tau_r = 1
t = np.arange(1, 100).astype(float)
pulse = np.zeros_like(t)
pulse[0] = 1
biexp = np.exp(-t / tau_d) - np.exp(-t / tau_r)
theta1, theta2 = tau2AR(tau_d, tau_r)
ar = np.zeros_like(t)
for i in range(len(t)):
    if i > 1:
        ar[i] = pulse[i] + theta1 * ar[i - 1] + theta2 * ar[i - 2]
    elif i > 0:
        ar[i] = pulse[i] + theta1 * ar[i - 1]
    else:
        ar[i] = pulse[i]
fig, ax = plt.subplots()
ax.plot(norm(biexp), label="biexp")
ax.plot(norm(ar), label="ar")
ax.legend()
print(theta1, theta2)
print(AR2tau(theta1, theta2))
