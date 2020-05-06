#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import subprocess

cpu_data = []
steps = 100
subprocess.run(["make","test-galaxy-cpu"])
# for i in [4096,8192,16384,32768,40960,49152]:
for i in [128,256,512,1024,2048,4096,8192]
	outputs = subprocess.run(["./bin/test-galaxy-cpu " + str(steps) + " " + str(i)], shell=True,capture_output=True,encoding='utf-8')
	tokens = outputs.stdout.split(", ")
	tokens[0] = float(tokens[0])
	tokens[1] = float(tokens[1])
	tokens[2] = float(tokens[2])
	tokens.append(i)
	cpu_data.append(tokens)
	print(i)



x = np.array(cpu_data)[:,3]
integrated_y = np.array(cpu_data)[:,1]
force_calc = np.array(cpu_data)[:,0]
total_time = np.array(cpu_data)[:,2]

fig,a =  plt.subplots(2,2)

a[0][0].plot(x,integrated_y)
a[0][0].set_title('Integration Time')
a[0][1].plot(x,force_calc)
a[0][1].set_title('Force Calc Time')
a[1][0].plot(x,total_time)
a[1][0].set_title('Total Time')
fig.tight_layout()
plt.savefig("scripts/cpu-galaxy-fig.jpg")




