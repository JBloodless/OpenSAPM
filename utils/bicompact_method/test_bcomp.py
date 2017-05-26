import numpy as np
from bicompact import bicompact_method

#a = const, signal is a "step"

a1=1
tau1=1
h1=1
x1_1 = np.arange(0,51,h1)
x0_1 = np.arange(0,51,h1)
for i in range(x1_1.size):
	x1_1[i] = 0
	x0_1[i] = 1
for i in range((x0_1.size)//3,2*(x0_1.size)//3):
        x0_1[i] = 3

x1_1 = bicompact_method(a1,tau1,h1,x0_1)
print(x0_1)
print(x1_1)


#a = const, signal is a "peak"

a2=1
tau2=1
h2=1
x1_2 = np.arange(0,51,h2)
x0_2 = np.arange(0,51,h2)
for i in range(x1_2.size):
	x1_2[i] = 0
for i in range(-25,25):
        x0_2[i+25] = -125*i*i+5*625

x1_2 = bicompact_method(a2,tau2,h2,x0_2)
print(x0_2)
print(x1_2)


#a = const, signal is a gaussian

a3=1
tau3=1
h3=1
x1_3 = np.arange(0,40,h3)
for i in range(x1_3.size):
	x1_3[i] = 0
x0_3 = np.array([0.5000,
0.5793,
0.6554,
0.7257,
0.7881,
0.8413,
0.8849,
0.9192,
0.9452,0.9641,0.9772,0.9861,0.9918,0.9953,0.9974,0.9987,0.9993,0.9997,0.9998,0.9999,0.9999,0.9998,0.9997,0.9993,0.9987,0.9974,0.9953,0.9918,0.9861,0.9772,0.9641,0.9452,0.9192,0.8849,0.8413,0.7881,0.7257,0.6554,0.5793,0.5000])

x1_3 = bicompact_method(a3,tau3,h3,x0_3)
print(x0_3)
print(x1_3)


"""
#a - "tear", signal is a peak

a4_1=0
a4_2=2
tau4=1
h4=1
x1_4_1 = np.arange(0,5,h4)
x1_4_2 = np.arange(0,5,h4)
for i in range(x1_4_1.size):
	x1_4_1[i] = 0
	x1_4_2[i] = 0
x0_4_1 = np.array([0, 1, 1, 3, 10])
x0_4_2 = np.array([10, 3, 1, 1, 0])

x1_4_1 = bicompact_method(a4_1,tau4,h4,x0_4_1,x1_4_1)
x1_4_2 = bicompact_method(a4_2,tau4,h4,x0_4_2,x1_4_2)
#x1_4 = x1_4_1 + x1_4_2
#x0_4 = x0_4_1 + x0_4_2
np.vstack((x0_4_1,x0_4_2))
print(np.hstack((x0_4_1,x0_4_2)))
print(np.hstack((x1_4_1,x1_4_2)))
"""

#a - "tear", signal is a peak

x1_4 = np.zeros(51)
h4=1
tau4=1
x1_4[0] = 1;
x0_4 = np.arange(0,51,h1)
for i in range(-25,25):
        x0_2[i+25] = -125*i*i+5*625
a4 = np.arange(51)
for i in range(a4.size):
	a4[i] = 1
for i in range((a4.size)//2,a4.size):
        a4[i] = 2
print(a4)

for i in range (x0_4.size-2):
	tmpx1 = np.copy(x1_4[i:i+3])
	tmp = bicompact_method(a4[i+1],tau4,h4,x0_4[i:i+3],tmpx1)
	#print(x1)
	#print(x0)
	x1_4[i+1] = tmp[0]

print(x0_4[1:x0_4.size-1])
print(x1_4[1:x0_4.size-1])

#a - "hat", signal is a peak

x1_5 = np.zeros(40)
h5=1
tau5=1
x1_5[0] = 1;
x0_5 = np.arange(0,40,h1)
for i in range(-25,25):
        x0_2[i+25] = -125*i*i+5*625
a5 = np.array([0.5000,
0.5793,
0.6554,
0.7257,
0.7881,
0.8413,
0.8849,
0.9192,
0.9452,0.9641,0.9772,0.9861,0.9918,0.9953,0.9974,0.9987,0.9993,0.9997,0.9998,0.9999,0.9999,0.9998,0.9997,0.9993,0.9987,0.9974,0.9953,0.9918,0.9861,0.9772,0.9641,0.9452,0.9192,0.8849,0.8413,0.7881,0.7257,0.6554,0.5793,0.5000])

print(a5)

for i in range (x0_5.size-2):
	tmpx1 = np.copy(x1_5[i:i+3])
	tmp = bicompact_method(a5[i+1],tau5,h5,x0_5[i:i+3],tmpx1)
	#print(x1)
	#print(x0)
	x1_5[i+1] = tmp[0]

print(x0_5[1:x0_5.size-1])
print(x1_5[1:x0_5.size-1])

#a - "tear", signal is a step

x1_6 = np.zeros(51)
h6=1
tau6=1
x1_6[0] = 1;
x0_6 = np.arange(0,51,h1)
for i in range(x0_6.size):
	x0_6[i] = 1
for i in range((x0_6.size)//3,2*(x0_6.size)//3):
        x0_6[i] = 3
a6 = np.arange(51)
for i in range(a4.size):
	a6[i] = 1
for i in range((a4.size)//2,a4.size):
        a6[i] = 2
print(a6)

for i in range (x0_6.size-2):
	tmpx1 = np.copy(x1_6[i:i+3])
	tmp = bicompact_method(a6[i+1],tau6,h6,x0_6[i:i+3],tmpx1)
	#print(x1)
	#print(x0)
	x1_6[i+1] = tmp[0]

print(x0_6[1:x0_6.size-1])
print(x1_6[1:x1_6.size-1])

#a - "hat", signal is a step

x1_7 = np.zeros(40)
h7=1
tau7=1
x1_7[0] = 1;
x0_7 = np.arange(0,40,h7)
for i in range(x0_7.size):
	x0_7[i] = 1
for i in range((x0_7.size)//3,2*(x0_7.size)//3):
        x0_7[i] = 3
#сорян за нижеследующее
a7 = np.array([0.5000,
0.5793,
0.6554,
0.7257,
0.7881,
0.8413,
0.8849,
0.9192,
0.9452,0.9641,0.9772,0.9861,0.9918,0.9953,0.9974,0.9987,0.9993,0.9997,0.9998,0.9999,0.9999,0.9998,0.9997,0.9993,0.9987,0.9974,0.9953,0.9918,0.9861,0.9772,0.9641,0.9452,0.9192,0.8849,0.8413,0.7881,0.7257,0.6554,0.5793,0.5000])

print(a7)

for i in range (x0_7.size-2):
	tmpx1 = np.copy(x1_5[i:i+3])
	tmp = bicompact_method(a7[i+1],tau7,h7,x0_7[i:i+3],tmpx1)
	#print(x1)
	#print(x0)
	x1_7[i+1] = tmp[0]

print(x0_7[1:x0_7.size-1])
print(x1_7[1:x1_7.size-1])

#drawing
import matplotlib.pyplot as plt
import numpy as np
import sys
import imageio
import os
import shutil
import glob

def draw1DSlice(solution, t_slice, x_start, x_end, legend, solution_max_value):
    M = len(solution)
    x_step = (x_end - x_start) / M
    x =  np.arange(x_start,x_end,x_step) 
    #Устанавливаем размеры рисунка.
    ax = plt.figure(figsize = (10,10)).add_subplot(111)
    #Устанавливаем подписи значений по осям.
    xax = ax.xaxis 
    xlabels = xax.get_ticklabels()
    for label in xlabels:
        label.set_fontsize(12)
    yax= ax.yaxis 
    ylabels = yax.get_ticklabels()
    for label in ylabels:
        label.set_fontsize(12)
    #Устанавливаем границы отображения по x,f(x). 
    plt.ylim(-3/2 * solution_max_value, 3/2 * solution_max_value)
    plt.xlim(x[0],x[-1])
        
    #Устанавливаем легенду на графике, включаем отображение сетки.
    plt.title(legend + ', t = ' + str(t_slice) + 's', fontsize = 14) 
    plt.xlabel('x', fontsize = 12)
    #plt.ylabel(legend+'(x)', fontsize = 12)
    plt.grid(True)
    
    plt.plot(x, solution, "--",linewidth=2)
    plt.show()

#for i in range (3):
#        draw1DSlice (zh[i], 1, 0, 10, 'a = const, signal is a "step"', max(zh[i]))
draw1DSlice (x1_1, tau1, 0, 50, 'a = const, signal is a "step"', max(x1_1))
draw1DSlice (x1_2, tau2, 0, 50, 'a = const, signal is a "peak"', max(x1_2))
draw1DSlice (x1_3, tau3, 0, 50, 'a = const, signal is a gaussian', max(x1_3))
draw1DSlice (x1_4[1:x1_4.size-1], tau4, 0, 50, 'a - "tear", signal is a peak', max(x1_4))
draw1DSlice (x1_5[1:x1_5.size-1], tau5, 0, 50, 'a - "hat", signal is a peak', max(x1_5))
draw1DSlice (x1_6[1:x1_6.size-1], tau6, 0, 50, 'a - "tear", signal is a step', max(x1_6))
draw1DSlice (x1_7[1:x1_7.size-1], tau7, 0, 50, 'a - "hat", signal is a step', max(x1_7))
