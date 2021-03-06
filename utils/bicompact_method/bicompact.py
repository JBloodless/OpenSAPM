import numpy as np
'''
Бикомпактный метод для уравнения переноса
a - параметр перед производной по x
tau - шаг по времени
h - шаг по координате
x0 - начальное условие t=0 (numpy array)
x1 - следующий слой с заданным граничным условием на левом или на правом конце (numpy array)
Возвращаемое значение: Следующий слой (x1) - numpy array
'''
def bicompactMethod(a,tau,h,x0,x1):

	# Так как работаем с целыми узлами - увеличиваем шаг вдвое
	h=2*h
	r=tau/h

	# Случай, когда краевый условия заданы на ЛЕВОМ конце и а>0
	if (a>0):

		# В данном случае итерируемся, начиная с левого края
		for j in range((x0.size-1)//2):
			M = np.array([[1/6+a*r,2/3],
						 [a*r+1/4,-2*r*a]])
			V = np.array([1/6*(x0[2*j]+x0[2*j+2]+4*x0[2*j+1])+(r*a-1/6)*x1[2*j],
							 1/4*(x0[2*j+2]-x0[2*j])+(1/4-r*a)*x1[2*j]])
			res = np.linalg.solve(M,V)
			x1[2*j+2] = res[0]
			x1[2*j+1] = res[1]

		# Если кол-во узлов - четное число, то необходима дополнительная итерация, т.к. метод - трехточечный
		if (x0.size%2==0):
			j = x0.size - 3;
			M = np.array([[1/6+a*r,2/3],
						 [a*r+1/4,-2*r*a]])
			V = np.array([1/6*(x0[j]+x0[j+2]+4*x0[j+1])+(r*a-1/6)*x1[j],
							 1/4*(x0[j+2]-x0[j])+(1/4-r*a)*x1[j]])
			res = np.linalg.solve(M,V)
			x1[j+2] = res[0]
			x1[j+1] = res[1]

	# Случай, когда краевый условия заданы на ПРАВОМ конце и a<0
	if (a<0):
		size = x0.size-1
		# В данном случае итерируемся, начиная с правого края
		for j in range((x0.size-1)//2):
			print(x1)
			print(x0)
			print()
			M = np.array([[1/6-a*r,2/3],
						 [a*r-1/4,-2*r*a]])
			V = np.array([1/6*(x0[size-2*j-2]+x0[size-2*j]+4*x0[size-2*j-1])+(-r*a-1/6)*x1[size-2*j],
							 1/4*(x0[size-2*j]-x0[size-2*j-2])+(-1/4-r*a)*x1[size-2*j]])
			res = np.linalg.solve(M,V)
			x1[size-2*j-2] = res[0]
			x1[size-2*j-1] = res[1]
		
		# Если кол-во узлов - четное число, то необходима дополнительная итерация, т.к. метод - трехточечный
		if (x0.size%2==0):
			j = 0;
			M = np.array([[1/6-a*r,2/3],
						 [a*r-1/4,-2*r*a]])
			V = np.array([1/6*(x0[j]+x0[j+2]+4*x0[j+1])+(-r*a-1/6)*x1[j+2],
							 1/4*(x0[j+2]-x0[j])+(-1/4-r*a)*x1[j+2]])
			res = np.linalg.solve(M,V)
			x1[j] = res[0]
			x1[j+1] = res[1]

	if (a==0):
		x1 = x0

	return x1[1:x1.size-1]

# tests

# a=-1
# tau=1
# h=1
# x1 = np.arange(0,9,h)
# for i in range(x1.size):
# 	x1[i] = 0
# x1[8]=9
# x0 = np.arange(0,9,h)

a=1
tau=1
h=1
x1 = np.arange(0,10,h)
for i in range(x1.size):
	x1[i] = 0
x0 = np.arange(0,10,h)

x1 = bicompactMethod(a,tau,h,x0,x1)
print(x1)
print(x0)