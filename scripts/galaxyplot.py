#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from subprocess import PIPE

cpu_data = []
gpu_data = []
steps = 10
subprocess.run(["make","test-galaxy-cpu"])
for i in [128,256,512,1024,2048,4096,8192]:
	outputs = subprocess.run(["./bin/test-galaxy-cpu " + str(steps) + " " + str(i)], shell=True,stdout=PIPE,encoding='utf-8')
	tokens = outputs.stdout.split(", ")
	tokens[0] = float(tokens[0])
	tokens[1] = float(tokens[1])
	tokens[2] = float(tokens[2])
	tokens[3] = float(tokens[3])
	tokens.append(i)
	cpu_data.append(tokens)
	print(i)

subprocess.run(["make","test-galaxy-gpu"])
for i in [128,256,512,1024,2048,4096,8192]:
	outputs = subprocess.run(["./bin/test-galaxy-gpu " + str(steps) + " " + str(i)], shell=True,stdout=PIPE,encoding='utf-8')
	tokens = outputs.stdout.split(", ")
	tokens[0] = float(tokens[0])
	tokens[1] = float(tokens[1])
	tokens[2] = float(tokens[2])
	tokens[3] = float(tokens[3])
	tokens.append(i)
	gpu_data.append(tokens)
	print(i)



x = np.array(cpu_data)[:,4]
cpu_precomp = np.array(cpu_data)[:,0]
cpu_integrated_y = np.array(cpu_data)[:,2]
cpu_force_calc = np.array(cpu_data)[:,1]
cpu_total_time = np.array(cpu_data)[:,3]

x = np.array(gpu_data)[:,4]
gpu_precomp = np.array(cpu_data)[:,0]
gpu_integrated_y = np.array(cpu_data)[:,2]
gpu_force_calc = np.array(cpu_data)[:,1]
gpu_total_time = np.array(cpu_data)[:,3]

plt.plot(x,cpu_precomp,label='CPU')
plt.plot(x,gpu_precomp,label='GPU')
plt.title('Precomputation Time')
plt.ylabel("Time (sec)")
plt.xlabel("Num Bodies")
plt.legend()
plt.tight_layout()
plt.savefig('plots/galaxy-fig-inte.jpg')
plt.close()

plt.plot(x,cpu_integrated_y,label='CPU')
plt.plot(x,gpu_integrated_y,label='GPU')
plt.title('Integration Time')
plt.ylabel("Time (sec)")
plt.xlabel("Num Bodies")
plt.legend()
plt.tight_layout()
plt.savefig('plots/galaxy-fig-inte.jpg')
plt.close()

plt.plot(x,cpu_force_calc,label='CPU')
plt.plot(x,gpu_force_calc,label='GPU')
plt.title('Force Calc Time')
plt.ylabel("Time (sec)")
plt.xlabel("Num Bodies")
plt.tight_layout()
plt.savefig('plots/galaxy-fig-force.jpg')
plt.close()

plt.plot(x,cpu_total_time,label='CPU')
plt.plot(x,gpu_total_time,label='GPU')
plt.title('Total Time')
plt.ylabel("Time (sec)")
plt.xlabel("Num Bodies")
plt.tight_layout()
plt.savefig("plots/galaxy-fig.jpg")




