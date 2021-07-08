import sys
import random
import SigmoidUnit

class Layer:
	#create a layer of n units, each of m inputs, learning rate l, initial random weights between -0.5 and 0.5
	def __init__(self,n,m,l):
		self._units = []
		for i in range(n):
			w = []
			for j in range(m):
				w.insert(j,random.random()-0.5)
			s = SigmoidUnit.SigmoidUnit(w,l)
			self._units.insert(i,s)
	
	def calculateOutputs(self,x):
		o = []
		for i in range(len(self._units)):
			o.insert(i,self._units[i].calculateOutput(x))
		return o
	
	def calculateErrorsOutputLayer(self,t):
		e = []
		for i in range(len(self._units)):
			e.insert(i,self._units[i].calculateErrorOutputLayer(t[i]))
		return e
	
	def calculateErrorsHiddenLayer(self,wkh,ek):
		e = []
		for i in range(len(self._units)):
			e.insert(i,self._units[i].calculateErrorHiddenLayer(wkh,ek))
		return e
	
def main():
	x = str(sys.argv[1])
	print("Testing the main() test client with command line arguments to test module.")
	print(x)
	l = Layer(2,2,0.5)
	for i in range(2):
		print(l._units[i])
	print(l.calculateOutputs([2,3]))

if __name__ == '__main__' : main()