import sys
import SigmoidUnit
import Layer

class Network:
	#create network of n layers
	#hidden layers of m units, output layer of l units
	def __init__(self,n,m,l,numnetworkinputs,learningrate):
		self._layers = []
		for i in range(n):
			#hidden layers
			if(i < n-1):
				hlayer = Layer.Layer(m,numnetworkinputs,learningrate)
				self._layers.insert(i,hlayer)
			#output layer
			else:
				olayer = Layer.Layer(l,m,learningrate)
				self._layers.insert(i,olayer)
			
	#propagate input forward through network
	def propagateForward(self,x):
		o = []
		for i in range(len(self._layers)):
			#first layer
			if(i == 0):
				o = self._layers[i].calculateOutputs(x)
			#hidden layers
			elif(i < len(self._layers)-1):
				o = self._layers[i].calculateOutputs(o)
			#output layer
			else:
				o = self._layers[i].calculateOutputs(o)
		return o
	
	#propagate errors backward through network
	def propagateErrorsBackward(self,t,ek,w):
		e = []
		j = 0
		i = len(self._layers)-1
		while i >= 0:
			#output layer
			if(i == len(self._layers)-1):
				e.insert(j,self._layers[i].calculateErrorsOutputLayer(t))
				j += 1
				
			#hidden layers
			else:
				
				e.insert(j,self._layers[i].calculateErrorsHiddenLayer( ,e[j-1]))
				j +=1

def main():
	x = str(sys.argv[1])
	print("Testing the main() test client with command line arguments to test module.")
	print(x)
	n = Network(2,3,1,2,0.5)
	for i in range(len(n._layers)):
		print(n._layers[i])
	print(n.propagateForward([2,3]))

if __name__ == '__main__' : main()