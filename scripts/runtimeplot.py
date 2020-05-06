import numpy as np
import matplotlib.pyplot as plt

x = [64, 128, 256, 384, 512, 640, 768, 896, 1024]

precomp = [[0.13, 1.1, 9.2, 31.6, 75.7, 149.3, 257.1, 404.8, 606.6], [0.17, 0.6, 2.1, 4.7, 8.7, 15.0, 26.2, 41.4, 62.8]]
comp = [[0.27, 0.9, 3.8, 10.2, 16.9, 27.3, 39.8, 54.8, 69.1], [1.95, 3.6, 7.3, 11.7, 17.2, 24.9, 35.3, 48.0, 63.7]]
total = [[0.4, 2.0, 13.0, 41.7, 92.6, 176.5, 296.9, 459.7, 675.7], [2.2, 4.3, 9.5, 16.5, 26.0, 39.9, 61.5, 89.6, 126.6]]

labels1 = ["CPU Precomputation Runtime ", "CPU Total Runtime"]
labels2 = ["GPU Precomputation Runtime ", "GPU Total Runtime"]
colors1 = ['#ff6600cc', '#0000ff99']
colors2 = ['#ff0000cc', '#00ff0099']

fig, ax = plt.subplots()
ax.stackplot(x, precomp[0], comp[0], labels=labels1, colors=colors1)
ax.stackplot(x, precomp[1], comp[1], labels=labels2, colors=colors2)
ax.legend(loc='upper left')
ax.xaxis.set_ticks(np.arange(0, 1024+1, 128))
plt.xlabel("Data Size")
plt.ylabel("Runtime (s)")
plt.show()


for i in range(2):
    labels = labels2
    colors = colors2
    if i == 0:
        labels = labels1
        colors = colors1

    fig, ax = plt.subplots()
    ax.stackplot(x, precomp[i], comp[i], labels=labels, colors=colors)
    ax.legend(loc='upper left')
    ax.xaxis.set_ticks(np.arange(0, 1024+1, 128))
    plt.xlabel("Data Size")
    plt.ylabel("Runtime (s)")
    plt.show()

fig, ax = plt.subplots(3)
ax[0].xaxis.set_ticks(np.arange(0, 1024+1, 128))
ax[0].plot(x,precomp[0],zorder=1, c="#0000ff99", label='Precomputation Runtime CPU') 
ax[0].scatter(x,precomp[0],zorder=2, c="#0000ff") 
ax[0].plot(x,precomp[1],zorder=1, c='#ff6600cc', label='Precomputation Runtime GPU') 
ax[0].scatter(x,precomp[1],zorder=2, c='#ff6600') 
ax[0].legend()

ax[1].xaxis.set_ticks(np.arange(0, 1024+1, 128))
ax[1].plot(x,comp[0],zorder=1, c="#0000ff99", label='Simulation Computation Runtime CPU') 
ax[1].scatter(x,comp[0],zorder=2, c="#0000ff") 
ax[1].plot(x,comp[1],zorder=1, c='#ff6600cc', label='Simulation Computation Runtime GPU') 
ax[1].scatter(x,comp[1],zorder=2, c='#ff6600') 
ax[1].legend()

ax[2].xaxis.set_ticks(np.arange(0, 1024+1, 128))
ax[2].plot(x,total[0],zorder=1, c="#0000ff99", label='Total Runtime CPU') 
ax[2].scatter(x,total[0],zorder=2, c="#0000ff") 
ax[2].plot(x,total[1],zorder=1, c='#ff6600cc', label='Total Runtime GPU') 
ax[2].scatter(x,total[1],zorder=2, c='#ff6600') 
ax[2].legend()

for axe in ax.flat:
    axe.set(xlabel="Data Size", ylabel='Runtime (s)')
plt.show()