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
	
	

def main():
	x = str(sys.argv[1])
	print("Testing the main() test client with command line arguments to test module.")
	print(x)
	l = Layer(2,2,0.5)
	for i in range(2):
		print(l._units[i])

if __name__ == '__main__' : main()