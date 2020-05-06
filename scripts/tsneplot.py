import numpy as np
import matplotlib.pyplot as plt

# script for plotting tsne embedding
f = open('output', 'r')
Lines = f.readlines()

data = []
for line in Lines:
    lineString = line.strip()
    tokens = lineString.split(", ")
    tokens[0] = float(tokens[0])
    tokens[1] = float(tokens[1])
    tokens[2] = int(tokens[2])
    data.append(tokens)

# Create data
x = np.array(data)[:, 0]
y = np.array(data)[:, 1]
colors = np.array(data)[:, 2]

# Plot
fig, ax = plt.subplots()

scatter = ax.scatter(x, y, c=colors, alpha=0.8)
leg = ax.legend(*scatter.legend_elements(),
                loc="lower left", title="Classes")
ax.add_artist(leg)

plt.xlabel('x')
plt.ylabel('y')
plt.savefig("output.png")
plt.clf()
