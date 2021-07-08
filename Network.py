import sys
import SigmoidUnit

class Network:
	def __init__():
		self._units = []
		

def main():
	x = str(sys.argv[1])
	print("Testing the main() test client with command line arguments to test module.")
	print(x)
	#print(linearCombinationInputs([1,2,3],[4,5,6]))
	s = SigmoidUnit([1,1],0.5)
	#print(s.linearCombinationInputs([2,3]))
	print(s.calculateOutput([2,3]))

if __name__ == '__main__' : main()