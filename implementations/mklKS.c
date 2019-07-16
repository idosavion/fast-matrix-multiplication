#include <time.h>
#include <stdio.h>
#include <math.h>
#include "mklKS.h"

#define LOOP_COUNT 3
static const int NUM_BLOCKS_REQUIRED = 6;
static const int BASE_CASE_DIM = 2;

int compare_ks_and_mkl(int dim)
{
	int i = 0, j = 0, n = 0;
	n = dim;

	//To handle when n is not power of k we do the padding with zero
	int pow = 1;
	while (pow < n)
	{
		pow = pow * 2;
	}
	n = pow;
	double* matrixA = (double*)mkl_malloc(pow * pow * sizeof(double), 64);
	double* matrixB = (double*)mkl_malloc(pow * pow * sizeof(double), 64);
	double* matrixC_naive = (double*)mkl_malloc(pow * pow * sizeof(double), 64);

	for (int i = 0; i < n * n; i += 2)
	{
		matrixA[i] = 0;
		matrixA[i + 1] = i;

		matrixB[i] = 0;
		matrixB[i + 1] = i + 1;
	}
	FILE* fp = fopen("log.txt", "a+");
	printf("Using dim = %d\n", dim);
	fprintf(fp, "Using dim = %d\n", dim);
	clock_t start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		matrixC_naive = mkl_multiplication(matrixA, matrixB, matrixC_naive, n);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	double msec = diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	//printf("Standard Multiplication took %.2f msecs\n", msec);
	fprintf(fp, "Standard Multiplication took % .2f msecs\n", msec);
	printf("Standard Multiplication took % .2f msecs\n", msec);

	unsigned mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, BASE_CASE_DIM, 1);
	double* work = (double*)MKL_malloc(mem * sizeof(double), 64);
	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, 1, work);
		ksBasisTransformRec(matrixB, n, 1, work);
		ksMultRec(matrixA, matrixB, matrixC_naive, n, 1, work);
		ksInverseBasisTransformRec(matrixC_naive, n, 1, work);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	MKL_free(work);
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("KS (1 step) took %.2f msecs\n", msec);
	fprintf(fp, "KS (1 step) took % .2f msecs\n", msec);


	mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, BASE_CASE_DIM, 2);
	work = (double*)MKL_malloc(mem * sizeof(double), 64);
	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, 2, work);
		ksBasisTransformRec(matrixB, n, 2, work);
		ksMultRec(matrixA, matrixB, matrixC_naive, n, 2, work);
		ksInverseBasisTransformRec(matrixC_naive, n, 2, work);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	MKL_free(work);
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("KS (2 step) took %.2f msecs\n", msec);
	fprintf(fp, "KS (2 step) took % .2f msecs\n", msec);

	mem = memoryRequirements(NUM_BLOCKS_REQUIRED, n, BASE_CASE_DIM, 3);
	work = (double*)MKL_malloc(mem * sizeof(double), 64);
	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		ksBasisTransformRec(matrixA, n, 3, work);
		ksBasisTransformRec(matrixB, n, 3, work);
		ksMultRec(matrixA, matrixB, matrixC_naive, n, 3, work);
		ksInverseBasisTransformRec(matrixC_naive, n, 3, work);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	MKL_free(work);
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("KS (3 step) took %.2f msecs\n", msec);
	fprintf(fp, "KS (3 step) took % .2f msecs\n", msec);

	// printf("\nStrassen's Multiplication Output:\n");
	//
	// strassensMultRec(matrixA, matrixB, matrixC_strassen, n, 1);
	// printf("Strassen");
	// printMatrix(matrixC_strassen, pow);
	// printf("naive");
	// printMatrix(matrixC_naive, pow);



	mkl_free(matrixA);
	mkl_free(matrixB);
	mkl_free(matrixC_naive);

	return 0;
}

static void S1_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, &matrixA[(ldA / 2) * ldA + (ldA / 2)], ldA, S, ldA / 2);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[(ldA / 2 + j) * ldA + (ldA / 2 + k)];
			}
		}
	}
}

static void T1_add(const double* matrixB, double* T, int ldB) {
	if (ldB <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldB / 2, ldB / 2, 1, &matrixB[(ldB / 2) * ldB + (ldB / 2)], ldB, T, ldB / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldB / 2; j++) {
			for (int k = 0; k < ldB / 2; k++) {
				T[j * (ldB / 2) + k] = matrixB[(ldB / 2 + j) * ldB + (ldB / 2 + k)];
			}
		}
	}
}


static void S2_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, &matrixA[(ldA / 2) * ldA], ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[(ldA / 2 + j) * ldA + k];
			}
		}
	}
}

static void T2_add(const double* matrixB, double* T, int ldB) {
	if (ldB <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldB / 2, ldB / 2, 1, &matrixB[(ldB / 2) * ldB], ldB, T, ldB / 2);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldB / 2; j++) {
			for (int k = 0; k < ldB / 2; k++) {
				T[j * (ldB / 2) + k] = matrixB[(ldB / 2 + j) * ldB + k];
			}
		}
	}
}

static void S3_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, &matrixA[ldA + (ldA / 2)], ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[j * ldA + (ldA / 2 + k)];
			}
		}
	}
}

static void T3_add(const double* matrixB, double* T, int ldB) {
	if (ldB <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldB / 2, ldB / 2, 1, &matrixB[ldB + (ldB / 2)], ldB, T, ldB / 2);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldB / 2; j++) {
			for (int k = 0; k < ldB / 2; k++) {
				T[j * (ldB / 2) + k] = matrixB[j * ldB + (ldB / 2 + k)];
			}
		}
	}
}

static void S4_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldA / 2, ldA / 2, 1, matrixA, ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[j * ldA + k];
			}
		}
	}
}

static void T4_add(const double* matrixB, double* T, int ldB) {
	if (ldB <= INTEL_MKL_THRESH) {
		MKL_Domatcopy('R', 'N', ldB / 2, ldB / 2, 1, matrixB, ldB, T, ldB / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldB / 2; j++) {
			for (int k = 0; k < ldB / 2; k++) {
				T[j * (ldB / 2) + k] = matrixB[j * ldB + k];
			}
		}
	}
}

static void S5_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, &matrixA[ldA / 2], ldA, -1, &matrixA[(ldA / 2) * ldA], ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[j * ldA + (ldA / 2 + k)] - matrixA[(j + ldA / 2) * ldA + k];
			}
		}
	}
}

static void T5_add(const double* matrixB, double* T, int ldB) {
	if (ldB <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldB / 2, ldB / 2, 1, &matrixB[(ldB / 2) * ldB + (ldB / 2)], ldB, -1, &matrixB[ldB / 2], ldB, T, ldB / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldB / 2; j++) {
			for (int k = 0; k < ldB / 2; k++) {
				T[j * (ldB / 2) + k] = matrixB[(ldB / 2 + j) * ldB + (ldB / 2 + k)] - matrixB[j * ldB + (ldB / 2 + k)];
			}
		}
	}
}

static void S6_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, &matrixA[ldA / 2], ldA, -1, matrixA, ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[j * ldA + (ldA / 2 + k)] - matrixA[j * ldA + k];
			}
		}
	}
}

static void T6_add(const double* matrixB, double* T, int ldB) {
	if (ldB <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldB / 2, ldB / 2, 1, &matrixB[ldB / 2], ldB, -1, &matrixB[(ldB / 2) * ldB], ldB, T, ldB / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldB / 2; j++) {
			for (int k = 0; k < ldB / 2; k++) {
				T[j * (ldB / 2) + k] = matrixB[j * ldB + (ldB / 2 + k)] - matrixB[(j + ldB / 2) * ldB + k];
			}
		}
	}
}

static void S7_add(const double* matrixA, double* S, int ldA) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, &matrixA[(ldA / 2) * ldA + (ldA / 2)], ldA, -1, &matrixA[ldA / 2], ldA, S, ldA / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				S[j * (ldA / 2) + k] = matrixA[(ldA / 2 + j) * ldA + (ldA / 2 + k)] - matrixA[j * ldA + (ldA / 2 + k)];
			}
		}
	}
}

static void T7_add(const double* matrixB, double* T, int ldB) {
	if (ldB <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldB / 2, ldB / 2, 1, &matrixB[ldB / 2], ldB, -1, matrixB, ldB, T, ldB / 2);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldB / 2; j++) {
			for (int k = 0; k < ldB / 2; k++) {
				T[j * (ldB / 2) + k] = matrixB[j * ldB + (ldB / 2 + k)] - matrixB[j * ldB + k];
			}
		}
	}
}

static void Q1_add(double* matrixA, int ldA, double* m4,double* m5) {
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, m4, ldA / 2, 1, m5, ldA / 2, matrixA, ldA);
}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				matrixA[j * ldA + k] = m4[j * ldA / 2 + k] + m5[j * ldA / 2 + k];
			}
		}
	}
}

static void Q2_addAddSubAdd(double* matrixA, int ldA, double* m3, double* m5, double* m6, double* m7) {
	// creating C12
	if (ldA <= INTEL_MKL_THRESH) {
		int size = ldA / 2 * ldA / 2;
		vdAdd(size, m3, m5, m3);
		vdSub(size, m3, m6, m3);
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, m3, ldA / 2, 1, m7, ldA / 2, &matrixA[ldA / 2], ldA);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				matrixA[j * ldA + (ldA / 2 + k)] = m3[j * ldA / 2 + k] + m5[j * ldA / 2 + k] - m6[j * ldA / 2 + k] + m7[j * ldA / 2 + k];
			}
		}
	}
}

static void Q3_add(double* matrixA, int ldA, double* m2, double* m7) {
	// creating C21
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, m2, ldA / 2, 1, m7, ldA / 2, &matrixA[(ldA / 2) * ldA], ldA);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				matrixA[(j + ldA / 2) * ldA + k] = m2[j * ldA / 2 + k] + m7[j * ldA / 2 + k];
			}
		}
	}
}

static void Q4_add(double* matrixA, int ldA, double* m1, double* m6) {
	// creating C22
	if (ldA <= INTEL_MKL_THRESH) {
		MKL_Domatadd('R', 'N', 'N', ldA / 2, ldA / 2, 1, m1, ldA / 2, -1, m6, ldA / 2, &matrixA[(ldA / 2) * ldA + (ldA / 2)], ldA);
	}
	else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int j = 0; j < ldA / 2; j++) {
			for (int k = 0; k < ldA / 2; k++) {
				matrixA[(j + ldA / 2) * ldA + (ldA / 2 + k)] = m1[j * ldA / 2 + k] + m6[j * ldA / 2 + k];
			}
		}
	}
}

/*
* Strassen's Multiplication algorithm using Divide and Conquer technique.
*/
void ksMultRec(const double* matrixA, const double* matrixB, double* matrixC, int n, int steps_left, double* work)
{
	if (steps_left == 0)
	{
		//This is the terminating condition for using strassen.
		mkl_multiplication(matrixA, matrixB, matrixC, n);
		return;
	}

	//Divide the matrix
	double* m1 = work;
	double* m2 = &m1[n / 2 * n / 2];
	double* m3 = &m1[2 * n / 2 * n / 2];
	double* m4 = &m1[3 * n / 2 * n / 2];
	//double* m5 = &m1[4 * n / 2 * n / 2];
	//double* m6 = &m1[5 * n / 2 * n / 2];
	//double* m7 = &m1[6 * n / 2 * n / 2];
	double* S = &m1[4 * n / 2 * n / 2];
	double* T = &m1[5 * n / 2 * n / 2];
	double* nextWork = &work[6 * n / 2 * n / 2];

	// m1 = 4th multiplication
	S4_add(matrixA, S, n);
	T4_add(matrixB, T, n);
	ksMultRec(S, T, m1, n / 2, steps_left - 1, nextWork);

	// m2 = 5th multiplication
	S5_add(matrixA, S, n);
	T5_add(matrixB, T, n);
	ksMultRec(S, T, m2, n / 2, steps_left - 1, nextWork);

	// creating C11
	// done with 4th multiplication, m1 can be reused
	Q1_add(matrixC, n, m1, m2);

	// m1 = 2nd multiplication
	S2_add(matrixA, S, n);
	T2_add(matrixB, T, n);
	ksMultRec(S, T, m1, n / 2, steps_left - 1, nextWork);

	// m3 = 7th multiplication
	S7_add(matrixA, S, n);
	T7_add(matrixB, T, n);
	ksMultRec(S, T, m3, n / 2, steps_left - 1, nextWork);

	// creating C21 2nd+7th
	// done with 2nd multiplication, m1 can be reused
	Q3_add(matrixC, n, m1, m3);
	//printMatrix(matrixC, n);

	// m1 = 1st multiplication
	S1_add(matrixA, S, n);
	T1_add(matrixB, T, n);
	ksMultRec(S, T, m1, n / 2, steps_left - 1, nextWork);
	
	// m4 = 6th multiplication
	S6_add(matrixA, S, n);
	T6_add(matrixB, T, n);
	ksMultRec(S, T, m4, n / 2, steps_left - 1, nextWork);

	// creating C22
	// done with 1st multiplication, m1 can be reused
	Q4_add(matrixC, n, m1, m4);

	// m1 = 3rd multiplication
	S3_add(matrixA, S, n);
	T3_add(matrixB, T, n);
	ksMultRec(S, T, m3, n / 2, steps_left - 1, nextWork);

	// creating C12
	Q2_addAddSubAdd(matrixC, n, m1, m2, m4, m3);
	//printMatrix(matrixC, n);

	return;
}
