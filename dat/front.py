import sys
import numpy as np
import matplotlib.pyplot as plt
#import yt
#from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper

fold = "./hydrogen/"

def interp(X, Y, i, x):
	return Y[i] + (Y[i+1]-Y[i])*(x-X[i])/(X[i+1]-X[i])


for arg in range(int(sys.argv[1]),int(sys.argv[2])):
	arr = np.fromfile("./hydrogen/"+"xhi%03d.bin" % arg, dtype=np.float32)
	temp = np.fromfile("./photons/"+"temp%03d.bin" % arg, dtype=np.float32)
	
	DIMX = int((arr.size+1)**(1.0/3.0))
	DIMX2 = int(DIMX/2)
	R = np.linspace(0.0, 6.6, DIMX)
	arr = 1.0 - arr.reshape(DIMX, DIMX, DIMX)
	temp = temp.reshape(DIMX, DIMX, DIMX)
	xHII = arr[0,0,:]
	T = temp[DIMX2,DIMX2,:]
	
	R0 = 0
	R1 = 0
	R2 = 0
	N0 = 0
	N1 = 0
	N2 = 0
	#print(xHII)
	for i in range(0,DIMX-1):
		if(xHII[i]>0.9 and xHII[i+1]<0.9):
			N0 = i
			R0 = interp(xHII, R, i, 0.9)*1.e3
			T0 = T[int(0.75*(i-DIMX/2)+DIMX/2)]
		if(xHII[i]>0.5 and xHII[i+1]<0.5):
			N1 = i
			R1 = interp(xHII, R, i, 0.5)*1.e3
			T1 = T[i]
		if(xHII[i]>0.1 and xHII[i+1]<0.1):
			N2 = i
			R2 = interp(xHII, R, i, 0.1)*1.e3
			T2 = T[i]
	av = 0
	for i in range(0, N0):
		av = av + 3*T[i]*((i-DIMX)**2)*1
	av = av/(N0-DIMX+0.0001)**3
	print (R0, R1, R2, N0, N1, N2, T0, T1, av)
	
	if(arg == 9):
		ax = plt.figure(num=None, figsize=(11,8.5), dpi=300)
		
		plt.plot(R, xHII, lw=3, label=r'$1.0-x_{HI}$', c='b')

		plt.axis([0,5,1.e-6,1.1e0])
		plt.legend(loc=1)
	#	plt.semilogy()
		plt.title(r'${\rm Hydrogen}\;{\rm Fraction}$', fontsize=24)
		plt.savefig('vis/profile.png')
		plt.close()
