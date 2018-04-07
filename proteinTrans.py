import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np 
from time import time

def genRNACodonTab(table_file):
	f = open(table_file, 'r')
	lines = f.readlines()
	f.close()
	rna_codon_tab = [line.split()[-1] for line in lines]
	return np.array(rna_codon_tab)

def proteinTrans(rnaSeqFile):
	#Prepare RNA seq as bytestring with tail
	f = open(rnaSeqFile, 'rb')
	rnaSeqText = f.read().upper()
	f.close()

	gridSize = 1024
	blockSize = cuda.device_attribute.MAX_THREADS_PER_BLOCK
	numbThreads = gridSize * blockSize

	rnaLen = len(rnaSeqText)
	q, r = divmod(rnaLen, numbThreads)
	q = q+1

	rnaSeq = np.frombuffer(rnaSeqText)
	rnaSeq = np.concatenate((rnaSeq, np.repeat('N', numbThreads - r)))
	numbSeq = np.zeros(rnaSeq.size).astype(np.int32)
	codonSeq = np.zeros(rnaSeq.size - 3 + 1).astype(np.int32)
	aaSeq = np.repeat('', codonSeq.size)

	#Get rna_codon_table
	rna_codon_table = genRNACodonTab("RNA_Codon_Tab.txt")

	#Init and load cuda kernels
	f = open("proteinTrans.cu", 'r')
	kernels = f.read()
	f.close()
	mod = SourceModule(kernels)
	mapToNumb = mod.get_function("mapToNumb")
	genNumbCodon = mod.get_function("genNumbCodon")
	mapToAA = mod.get_function("mapToAA")

	#Run cuda kernels
	N = numbThreads
	M = q
	mapToNumb(np.int32(N), np.int32(M), cuda.In(rnaSeq), cuda.Out(numbSeq), grid=(gridSize, 1, 1), block=(blockSize, 1, 1))
	genNumbCodon(np.int32(N), np.int32(M), cuda.In(numbSeq), cuda.Out(codonSeq), grid=(gridSize, 1, 1,), block=(blockSize, 1, 1))
	mapToAA(np.int32(N), np.int32(M), cuda.In(rna_codon_table), cuda.In(codonSeq), cuda.Out(aaSeq), grid=(gridSize, 1, 1), block=(blockSize, 1, 1))

	return aaSeq

def main():
	rnaFile = "rna_seq.txt"
	aaSeq = proteinTrans(rnaFile)
	print(aaSeq)

if __name__=="__main__":
	main()