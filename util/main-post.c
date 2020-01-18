/**
   Copyright (c) 2014-2015, Sandia Corporation
   All rights reserved.

   This file is part of fast-matmul and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause.
*/
/*#include "linalg.hpp"
#include "algorithms/strassen.hpp"
#include "timing.hpp"

#include <iostream>

int main(int argc, char** argv) {
	int m = 8192;
	int k = 8192;
	int n = 8192;
	int numsteps = 4;

	Matrix<double> A = RandomMatrix<double>(m, k);
	Matrix<double> B = RandomMatrix<double>(k, n);
	Matrix<double> C1(m, n), C2(m, n);

	//mkl_set_num_threads(2);

	Time([&] { MatMul(A, B, C1); }, "Classical gemm");
	double time = strassen::FastMatmul(A, B, C2, numsteps);
	std::cout << "Fast time: " << time << " ms" << std::endl;

	// Test for correctness.
	std::cout << "Maximum relative difference: " << MaxRelativeDiff(C1, C2) << std::endl;

	return 0;
}*/

#include "main.h"
#include "mkl.h"
#include <stdint.h>
#include <stdbool.h> 
#include <time.h>

static uint64_t mat_rng[2] = { 11ULL, 1181783497276652981ULL };

static inline uint64_t xorshift128plus(uint64_t s[2])
{
	uint64_t x, y;
	x = s[0], y = s[1];
	s[0] = y;
	x ^= x << 23;
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
	y += s[1];
	return y;
}

double mat_drand(void)
{
	return (xorshift128plus(mat_rng) >> 11) * (1.0 / 9007199254740992.0);
}

void printMatix(double* mat, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%f ", mat[i * cols + j]);
		}
		printf("\n");
	}

}

void cacheCleanLoop(int i) {
	double* matrixtmp = (double*)mkl_malloc(i * i * sizeof(double), 64);
	for (int j = 0; j < i * i; j++)
	{
		matrixtmp[j] = mat_drand();
	}
	mkl_free(matrixtmp);
}

int main(int argc, char** argv)
{
	bool loop = true;
	bool vml = false;
	bool hybridLoop = false;
	bool blockAdd = false;
	int i = 2048*2*2*2*2;

	/*while (i == 2*2048)
	{
		printf("Using dim = %d\n", i);
		compare_strassen_and_mkl(i);
		i = i * 2;
	}*/

	//To handle when n is not power of k we do the padding with zero
	int pow = i/2;

	double* matrixA = (double*) mkl_malloc(pow * pow * sizeof(double), 64);
	double* matrixC = (double*)mkl_malloc(pow * pow * sizeof(double), 64);
	double* matrixB = (double*)mkl_malloc(pow * pow * sizeof(double), 64);

	for (int j = 0; j < pow * pow; j ++)
	{
		matrixA[j] = mat_drand();
		matrixB[j] = mat_drand();
		matrixC[j] = mat_drand();
	}
	cacheCleanLoop(pow);

	double* matrixC_strassen = (double*)mkl_malloc(pow * pow * sizeof(double), 64);
	//memset(matrixC_strassen, 0, pow / 2 * pow / 2 * sizeof(double));
	
	double msec;
	clock_t start, diff;
# if defined(_OPENMP)
	printf("OPENMP \n");
	printf("%d \n", omp_get_thread_num());
# endif
	start = clock();
	if (loop == true) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < pow; j++) {
			for (int k = 0; k < pow; k++) {
				matrixC_strassen[j * pow + k] = matrixA[j * pow + k] + matrixB[pow*j + k] -matrixC[pow*j + k];
			}
		}
		//diff = clock() - start;
		msec = ((double)((clock() - start)*1000) / CLOCKS_PER_SEC); //diff * 1000 / (CLOCKS_PER_SEC);
		printf("LOOP: %.2f msecs %d \n", msec, i);
	}

	start = clock();
	if (hybridLoop == true) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < i; j++) {
			vdAdd(pow, &matrixA[pow*j], &matrixB[pow*j], &matrixC_strassen[j * pow]);
			vdSub(pow, &matrixC_strassen[pow *j], &matrixC[pow * j], &matrixC_strassen[pow *j]);
			//for (int k = 0; k < i / 2; k++) {
			//	matrixC_strassen[j * (i / 2) + k] = matrixA[j * i + k] + matrixA[(i / 2 + j) * i + (i / 2 + k)] - matrixA[j * i + (i / 2 + k)];
			//}
		}
		//diff = clock() - start;
		msec = ((double)((clock() - start) * 1000) / CLOCKS_PER_SEC); //diff * 1000 / (CLOCKS_PER_SEC);
		printf("HYBRIDLOOP: %.2f msecs %d \n", msec, i);
	};
	
	
	//memset(matrixB, 0, pow / 2 * pow / 2 * sizeof(double));
	clock_t start1, end, msec1;
	if (vml == true) {
		//double* matrixB = (double*)mkl_malloc(pow / 2 * pow / 2 * sizeof(double), 64);
		//double* matrixC = (double*)mkl_malloc(pow / 2 * pow / 2 * sizeof(double), 64);
		//double* matrixD = (double*)mkl_malloc(pow / 2 * pow / 2 * sizeof(double), 64);
		start = clock();
		//mkl_domatcopy('R', 'N', i / 2, i / 2, 1, &matrixA[i * i / 2 + i / 2], i, matrixB, i / 2);
		//mkl_domatcopy('R', 'N', i / 2, i / 2, 1, &matrixA[i / 2], i, matrixC, i / 2);
		//mkl_domatcopy('R', 'N', i / 2, i / 2, 1, matrixA, i, matrixD, i / 2);
		start1 = clock();
		vdAdd(pow * pow, matrixA, matrixB, matrixC_strassen);
		vdSub(pow * pow, matrixC_strassen, matrixC, matrixC_strassen);
		end = clock();
		msec = ((double)((end - start)*1000) / CLOCKS_PER_SEC);
		msec1 = ((double)((end - start1)*1000) / CLOCKS_PER_SEC);
		printf("VML: %.2f msecs %d \nVMLADD: %.2f msecs \n ", msec, i, msec1);
	}

	if (blockAdd == true) {
		start = clock();
		mkl_domatadd('R', 'N', 'N', pow, pow, 1, matrixA, pow, 1, matrixB, pow, matrixC_strassen, pow);
		mkl_domatadd('R', 'N', 'N', pow, pow, 1, matrixC_strassen, pow, -1, matrixC, pow, matrixC_strassen, pow);
		end = clock();
		msec = ((double)((end - start) * 1000) / CLOCKS_PER_SEC);
		printf("MATADD: %.2f msecs %d \n ", msec, i);
	}
	
	/*mkl_free(matrixC_naive); 
	mkl_free(matrixC_strassen);
	mkl_free(matrixA);
	mkl_free(matrixB);
	mkl_free(matrixC);*/
	return 0;
}
