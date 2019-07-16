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
#if defined(_OPENMP)
#include "omp.h"
#endif
#define __ELAYE_SSE__ 1

#ifdef __ELAYE_SSE__
#include <immintrin.h>
#endif // __ELAYE_SSE__


static uint64_t mat_rng[2] = { 11ULL, 1181783497276652981ULL };

enum bench {
	ksBasis = 2,
	ks = 3,
	strassen = 0,
	winograd = 4,
};

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

void cacheCleanLoop(int k) {
	int i = 8192;
	double* matrixtmp = (double*)mkl_malloc(i * i * sizeof(double), 64);
	for (int j = 0; j < i * i; j++)
	{
		matrixtmp[j] = mat_drand();
	}
	mkl_free(matrixtmp);
}

void vml_pre(const double* matrixA, double* matrixOut, int i) {
	clock_t start, start1, end; 
	double msec, msec1;
	int pow = i;
	double* matrixB = (double*)mkl_malloc(pow / 2 * pow / 2 * sizeof(double), 64);
	double* matrixC = (double*)mkl_malloc(pow / 2 * pow / 2 * sizeof(double), 64);
	double* matrixD = (double*)mkl_malloc(pow / 2 * pow / 2 * sizeof(double), 64);
	start = clock();
	mkl_domatcopy('R', 'N', i / 2, i / 2, 1, &matrixA[i * i / 2 + i / 2], i, matrixB, i / 2);
	mkl_domatcopy('R', 'N', i / 2, i / 2, 1, &matrixA[i / 2], i, matrixC, i / 2);
	mkl_domatcopy('R', 'N', i / 2, i / 2, 1, matrixA, i, matrixD, i / 2);
	start1 = clock();
	vdAdd(i, matrixC, matrixB, matrixOut);
	vdSub(i, matrixOut, matrixC, matrixOut);
	end = clock();
	msec = ((double)((end - start) * 1000) / CLOCKS_PER_SEC);
	msec1 = ((double)((end - start1) * 1000) / CLOCKS_PER_SEC);
	printf("VML: %.2f msecs %d  VMLADD: %.2f msecs \n ", msec, i, msec1);
	mkl_free(matrixB); mkl_free(matrixC); mkl_free(matrixD);
}

void blockAdd_pre(const double* matrixA, double* matrixC, int i) {
	clock_t start, end;
	double msec;
		start = clock();
	mkl_domatadd('R', 'N', 'N', i / 2, i / 2, 1, &matrixA[i * i / 2 + i / 2], i, 1, &matrixA[i / 2], i, matrixC, i / 2);
	mkl_domatadd('R', 'N', 'N', i / 2, i / 2, 1, matrixC, i / 2, -1, matrixA, i, matrixC, i / 2);
	end = clock();
	msec = ((double)((end - start) * 1000) / CLOCKS_PER_SEC);
	printf("MATADD: %.2f msecs %d \n ", msec, i);
}

void hybridLoop_pre(const double* matrixA, double* matrixC, int i) {
	clock_t start;
	double msec;
	start = clock();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < i / 2; j++) {
		vdAdd(i / 2, &matrixA[j * i], &matrixA[i * (j + i / 2) + i / 2], &matrixC[j * (i / 2)]);
		vdSub(i / 2, &matrixC[j * (i / 2)], &matrixA[i * j + i / 2], &matrixC[j * (i / 2)]);
		//for (int k = 0; k < i / 2; k++) {
		//	matrixC_strassen[j * (i / 2) + k] = matrixA[j * i + k] + matrixA[(i / 2 + j) * i + (i / 2 + k)] - matrixA[j * i + (i / 2 + k)];
		//}
	}
	//diff = clock() - start;
	msec = ((double)((clock() - start) * 1000) / CLOCKS_PER_SEC); //diff * 1000 / (CLOCKS_PER_SEC);
	printf("HYBRIDLOOP: %.2f msecs %d \n", msec, i);
}

void loop_pre(const double* matrixA, double* matrixC, int i) {
	clock_t start;
	double msec;
	start = clock();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < i / 2; j++) {
		for (int k = 0; k < i / 2; k++) {
			matrixC[j * (i / 2) + k] = matrixA[j * i + k] + matrixA[(i / 2 + j) * i + (i / 2 + k)] - matrixA[j * i + (i / 2 + k)];
		}
	}
	//diff = clock() - start;
	msec = ((double)((clock() - start) * 1000) / CLOCKS_PER_SEC); //diff * 1000 / (CLOCKS_PER_SEC);
	printf("LOOP: %.2f msecs %d \n", msec, i);
}

#ifdef __ELAYE_SSE__
inline void addToDoubleVectorSSE(const double* what, const double* toWhat, volatile double* dest, const unsigned int len)
{
	__m128d* _what = (__m128d*)what;
	__m128d* _toWhat = (__m128d*)toWhat;
	__m128d* _toWhatBase = (__m128d*)toWhat;

	__m128d _dest1;
	__m128d _dest2;

#ifdef FAST_SSE
	for (register unsigned int i = 0; i < len; i += 4, _what += 2, _toWhat += 2, _toWhatBase += 2)
	{
		_toWhatBase = _toWhat;
		_dest1 = _mm_add_pd(*_what, *_toWhat);               //line A
		_dest2 = _mm_add_pd(*(_what + 1), *(_toWhat + 1));    //line B

		*_toWhatBase = _dest1;
		*(_toWhatBase + 1) = _dest2;
	}
#else
	for (register unsigned int i = 0; i < len; i += 4)
	{
		_toWhatBase = _toWhat;
		_dest1 = _mm_add_pd(*_what++, *_toWhat++);
		_dest2 = _mm_add_pd(*_what++, *_toWhat++);

		*_toWhatBase++ = _dest1;
		*_toWhatBase++ = _dest2;
	}
#endif
}

void asm_tst(double* matrixA, int i) {
	for (int j = 0; j < i / 2; j++) {
		__m256d* _A1 = (__m256d*) & matrixA[j * i];
		for (register unsigned int k = 0; k < i / 2; k += 4, _A1 += 1)
		{
			double* x = (double*)_A1;
			printf("%f %f %f %f", *x, *(x+1), *(x + 2), *(x + 3));
		}
		printf("\n");
	}
}

void asm_loop(double const* matrixA, double* matrixC, int i) {
	clock_t start;
	double msec;
	__m256d _dest;
	start = clock();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < i / 2; j++) {
		__m256d* _A1 = (__m256d*) &matrixA[j * i];
		__m256d* _A2 = (__m256d*) &matrixA[i * (j + i / 2) + i / 2];
		__m256d* _A3 = (__m256d*) &matrixA[i * j + i / 2];
		__m256d* _C = (__m256d*) &matrixC[j * (i / 2)];
		for (register unsigned int k = 0; k < i/2; k += 4, _A1 += 1, _A2 += 1, _A3 += 1, _C += 1)
		{
			__m256d a1 = _mm256_load_pd(_A1);
			__m256d a2 = _mm256_load_pd(_A2);
			__m256d a3 = _mm256_load_pd(_A3);
			_dest = _mm256_add_pd(a1, a2); 
			*_C = _mm256_sub_pd(_dest, a3);
		}
	}
	//diff = clock() - start;
	msec = ((double)((clock() - start) * 1000) / CLOCKS_PER_SEC); //diff * 1000 / (CLOCKS_PER_SEC);
	printf("ASM HYBRID: %.2f msecs %d \n", msec, i);
}
#endif

int main(int argc, char** argv)
{
	bool loop = false;
	bool vml = false;
	bool hybridLoop = false;
	bool blockAdd = false;
	bool sse = true;
	bool benchmark = false;
	enum bench which = ks;
	int i = 8;

	while (i <= 2*2*2*2048 && benchmark)
	{
		printf("Using dim = %d\n", i);
		switch (which)
		{
		case ksBasis:
			benchmarkKSBasisTransform(i);
			break;
		case ks:
			compare_ks_and_mkl(i);
			break;
		case strassen:
			compare_strassen_and_mkl(i);
			break;
		case winograd:
			break;
		default:
			break;
		}
		//break;
		i = i * 2;
	}

	if (benchmark) {
		return 0;
	}

	//To handle when n is not power of k we do the padding with zero
	int pow = i;

	//memset(matrixC_strassen, 0, pow / 2 * pow / 2 * sizeof(double));
	while (pow <= 32768) {
	//while (pow <= 1024) {
		double* matrixA = (double*)mkl_malloc(pow * pow * sizeof(double), 64);
		double* matrixC_strassen = (double*)mkl_malloc(pow / 2 * pow / 2 * sizeof(double), 64);
		for (int j = 0; j < pow * pow; j++)
		{
			matrixA[j] = mat_drand();
			//matrixB[j] = mat_drand();
			//matrixC[j] = mat_drand();
		}
		cacheCleanLoop(pow);
		if (loop) {
			loop_pre(matrixA, matrixC_strassen, pow);
		}
		else if (vml) {
			vml_pre(matrixA, matrixC_strassen, pow);
		}
		else if (hybridLoop) {
			hybridLoop_pre(matrixA, matrixC_strassen, pow);
		}
		else if (blockAdd) {
			blockAdd_pre(matrixA, matrixC_strassen, pow);
		}
#ifdef __ELAYE_SSE__
		if (sse) {
			asm_loop(matrixA, matrixC_strassen, pow);
		}
#endif
		pow *= 2;
		mkl_free(matrixA);
		mkl_free(matrixC_strassen);
	}

	return 0;
}
