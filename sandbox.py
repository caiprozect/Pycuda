import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np 
from collections import Counter

def main():
	#fname = "~/Downloads/'Bioinformatics Python'/data/hg38.chroms/chr1.fa"
	fname = "E.ColiGenome.txt" #Toy example
	K = 1
	f = open(fname, 'r')
	#f.readline()
	data = f.read().upper().splitlines()
	f.close()
	data = ''.join(data).encode('utf-8')
	h_seq = np.frombuffer(data, dtype=np.uint8)
	h_seq = h_seq.astype(np.int32)

	mod = SourceModule("""
		__global__ void mapToNumb(
			const int N,
			const int M,
			const int numbKmer,
			int* seq,
			int* numb_seq
		)
		{
			int gid = blockDim.x * blockIdx.x + threadIdx.x;
			int idx = gid * M;
			int i, letter;

			if(idx < N*M) {
			for(i=0; i < M; i++) {
				letter = seq[idx+i];
				if(letter == 65) {
					numb_seq[idx+i] = 0;
				} else {
				if(letter == 67) {
					numb_seq[idx+i] = 1;
				} else {
				if(letter == 71) {
					numb_seq[idx+i] = 2;
				} else {
				if(letter == 84) {
					numb_seq[idx+i] = 3;
				} else {
					numb_seq[idx+i] = (-1) * numbKmer;
				}
				}
				}
				}
			}
		}
		}
		__global__ void freqTab(
		const int N,
		const int M,
		const int nK,
		const int numbKmer,
		int* numb_seq,
		int* freqs
	) {
		int gid = blockDim.x * blockIdx.x + threadIdx.x;
		int idx = gid * M;
		int i, numb;
		int k, loc_idx, ptn_idx;
		int dgt;
		for(i=0; i < M; i++) {
			ptn_idx = 0;
			loc_idx = idx + i;
			if(loc_idx < (N*M - nK + 1)) {
				for(k=0; k < nK; k++) {
					if((nK-1-k)==0) {
						dgt = 1;
					} else {
						dgt = (int)(powf(4, (float)(nK-1-k)));
					}
					numb = numb_seq[loc_idx + k];
					ptn_idx += dgt * numb;
				}
				if(ptn_idx >= 0){
				atomicAdd(&freqs[ptn_idx], 1);
				}
			}
		}
	}
		""")

	gridSize = 1024
	blockSize = 1024

	seqLen = h_seq.size
	q, r = divmod(seqLen, gridSize*blockSize)
	q = q + 1
	h_seq = np.concatenate((h_seq, np.zeros(gridSize*blockSize-r).astype(np.int32)))

	h_numb_seq = np.repeat(-1, h_seq.size)
	print(q)
	print(r)

	N = gridSize * blockSize
	M = q
	numbKmer = 4**K

	mapToNumb = mod.get_function("mapToNumb")
	mapToNumb(np.int32(N), np.int32(M), np.int32(numbKmer), cuda.In(h_seq), cuda.Out(h_numb_seq), block=(blockSize, 1, 1), grid=(gridSize, 1))

	print(h_numb_seq)

	h_freqs = np.zeros(numbKmer).astype(np.int32)

	freqTab = mod.get_function("freqTab")
	freqTab(np.int32(N), np.int32(M), np.int32(K), np.int32(numbKmer), cuda.In(h_numb_seq), cuda.Out(h_freqs), block=(blockSize, 1, 1), grid=(gridSize, 1))

	print(h_freqs)

if __name__=="__main__":
	main()