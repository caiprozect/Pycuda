__global__ void mapToNumb(
	const int N, //Number of whole threads
	const int M, //Length of subseq that one thread handles
	char* seq, 
	int* numb_seq
)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = gid * M;
	int i, letter;

	if(idx < N*M) {
	for(i=0; i < M; i++) {
		letter = seq[idx+i];
		if(letter == 'A') {
			numb_seq[idx+i] = 0;
		} else {
		if(letter == 'C') {
			numb_seq[idx+i] = 1;
		} else {
		if(letter == 'G') {
			numb_seq[idx+i] = 2;
		} else {
		if(letter == 'U') {
			numb_seq[idx+i] = 3;
		} else {
			numb_seq[idx+i] = (-1) * (int)(powf(4, (float)3));
		}
		}
		}
		}
	}
	}
}

__global__ void genNumbCodon(
	const int N,
	const int M,
	int* numb_seq,
	int* codon_seq
	)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = gid * M;
	int i, k;
	int codon_numb, loc_idx, numb, base;

	for(i=0; i < M; i++) {
		codon_numb = 0;
		loc_idx = idx + i;
		if(loc_idx <= N*M -3 + 1) {
			for(k=0; k<3; k++) {
				numb = numb_seq[loc_idx];
				base = (int)powf(4, (float)(2-k));
				codon_numb += numb * base;
			}
			codon_seq[loc_idx] = codon_numb;
		}
	}	
}

__global__ void mapToAA(
	const int N,
	const int M,
	char* rna_codon_tab,
	int* codon_seq,
	char* aa_seq
	)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = gid * M;
	int codon_idx, loc_idx;
	int i;
	
	for(i=0; i < M; i++) {
		loc_idx = idx + i;
		codon_idx = codon_seq[loc_idx];
		if(loc_idx <= N*M -3 + 1) {
			if(codon_idx >= 0) {
				aa_seq[loc_idx] = rna_codon_tab[codon_idx];
			}
		}
	}

}















