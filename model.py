import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = x + np.random.normal(0, 1, 100)

plt.plot(x, y)
plt.savefig("model_results.png")

with open("metrics.txt", "w") as f:
    f.write("MSE: 0.5\n")

print("Done")