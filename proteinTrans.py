import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compliler import SourceModule

import numpy as np 
from time import time

def genRNACodonTab(table_file):
	f = open(table_file, 'r'):
	lines = f.readlines()
	aas = [line.split()[-1] for line in lines]
	return np.array(aas).astype(uint8)



def main():

if __name__=="__main__":
	main()