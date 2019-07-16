#include <time.h>
#include <stdio.h>
#include <math.h>
#include "mkl_strassen.h"
#define LOOP_COUNT 3

int compare_strassen_and_mkl(int dim)
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
		matrixC_naive[i] = 0;
	}

	clock_t start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		matrixC_naive = mkl_multiplication(matrixA, matrixB, matrixC_naive, n);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	double msec = diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Standard Multiplication took %.2f msecs\n", msec);

	unsigned mem = memoryRequirements(9, n, 2, 1);
	start = clock(), diff;
	double* work = (double*)MKL_malloc(mem * sizeof(double), 64);
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		matrixC_naive = strassensMultRec(matrixA, matrixB, matrixC_naive, n, 1, work);
	}
	// printMatrix(matrixC_naive, pow);
	MKL_free(work);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Strassen Multiplication (1 step) took %.2f msecs\n", msec);


	mem = memoryRequirements(9, n, 2, 2);
	start = clock(), diff;
	work = (double*)MKL_malloc(mem * sizeof(double), 64);
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		matrixC_naive = strassensMultRec(matrixA, matrixB, matrixC_naive, n, 2, work);
	}
	// printMatrix(matrixC_naive, pow);
	MKL_free(work);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Strassen Multiplication (2 step) took %.2f msecs\n", msec);

	mem = memoryRequirements(9, n, 2, 3);
	start = clock(), diff;
	work = (double*)MKL_malloc(mem * sizeof(double), 64);
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		matrixC_naive = strassensMultRec(matrixA, matrixB, matrixC_naive, n, 3, work);
	}
	// printMatrix(matrixC_naive, pow);
	MKL_free(work);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Strassen Multiplication (3 step) took %.2f msecs\n", msec);

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

/*
* Wrapper function over strassensMultRec.
*/
double* strassensMultiplication(const double* matrixA, const double* matrixB, double* matrixC, int n, int steps_left, double* work)
{
	double* result = strassensMultRec(matrixA, matrixB, matrixC, n, steps_left, work);
	return result;
}

static void S1_add(const double* matrixA, double* S, int ldA) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldA / 2; j++) {
		for (int k = 0; k < ldA / 2; k++) {
			S[j * (ldA / 2) + k] = matrixA[j * ldA + k] + matrixA[j * ldA k];
		}
	}
}

static void T1_add(const double* matrixB, double* T, int ldB) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldB / 2; j++) {
		for (int k = 0; k < ldB / 2; k++) {
			T[j * (ldB / 2) + k] = matrixB[j * ldB + k];
		}
	}
}


static void S2_add(const double* matrixA, double* S, int ldA) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldA / 2; j++) {
		for (int k = 0; k < ldA / 2; k++) {
			S[j * (ldA / 2) + k] = matrixA[j * ldA + (ldA / 2 + k)];
		}
	}
}

static void T2_add(const double* matrixB, double* T, int ldB) {
	//TODO: replace with domatcopy?
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldB / 2; j++) {
		for (int k = 0; k < ldB / 2; k++) {
			T[j * (ldB / 2) + k] = matrixB[(ldB/2 + j) * ldB + k];
		}
	}
}

static void S3_add(const double* matrixA, double* S, int ldA) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldA / 2; j++) {
		for (int k = 0; k < ldA / 2; k++) {
			S[j * (ldA / 2) + k] = matrixA[(ldA/2 + j) * ldA + (ldA/2 + k)] + matrixA[(ldA / 2 + j) * ldA + k];
		}
	}
}

static void T3_add(const double* matrixB, double* T, int ldB) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldB / 2; j++) {
		for (int k = 0; k < ldB / 2; k++) {
			T[j * (ldB / 2) + k] = matrixB[j * ldB + (ldB / 2 + k)] - matrixB[j * ldB + k];
		}
	}
}

static void S4_add(const double* matrixA, const double* S3, double* S, int ldA) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldA / 2; j++) {
		for (int k = 0; k < ldA / 2; k++) {
			S[j * (ldA / 2) + k] = S3[(j * ldA/2) + k] - matrixA[j * ldA + k];
		}
	}
}

static void T4_add(const double* matrixB, const double* T3, double* T, int ldB) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldB / 2; j++) {
		for (int k = 0; k < ldB / 2; k++) {
			T[j * (ldB / 2) + k] = matrixB[(ldB / 2 + j) * ldB + (ldB/2 + k)] - T3[(j * ldB /2 ) + k];
		}
	}
}

static void S5_add(const double* matrixA, double* S, int ldA) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldA / 2; j++) {
		for (int k = 0; k < ldA / 2; k++) {
			S[j * (ldA / 2) + k] = matrixA[j * ldA + k] - matrixA[(j + ldA/2) * ldA + k];
		}
	}
}

static void T5_add(const double* matrixB, double* T, int ldB) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldB / 2; j++) {
		for (int k = 0; k < ldB / 2; k++) {
			T[j * (ldB / 2) + k] = matrixB[(j + ldB/2) * ldB + (ldB/2 + k)];
		}
	}
}

static void S6_add(const double* matrixA, const double* S4, double* S, int ldA) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldA / 2; j++) {
		for (int k = 0; k < ldA / 2; k++) {
			S[j * (ldA / 2) + k] = matrixA[ j * ldA + (ldA / 2 + k)] - S4[(j * ldA / 2) + k];
		}
	}
}

static void T6_add(const double* matrixB, double* T, int ldB) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldB / 2; j++) {
		for (int k = 0; k < ldB / 2; k++) {
			T[j * (ldB / 2) + k] = matrixB[(j + ldB / 2) * ldB + (ldB / 2 + k)];
		}
	}
}

static void S7_add(const double* matrixA, double* S, int ldA) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldA / 2; j++) {
		for (int k = 0; k < ldA / 2; k++) {
			S[j * (ldA / 2) + k] = matrixA[(ldA / 2 + j) * ldA + (ldA / 2 + k)];
		}
	}
}

static void T7_add(const double* matrixB, double* T, int ldB) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldB / 2; j++) {
		for (int k = 0; k < ldB / 2; k++) {
			T[j * (ldB / 2) + k] = matrixB[(ldB / 2 + j) * ldB + k] + matrixB[(ldB / 2 + j) * ldB + (ldB / 2 + k)];
		}
	}
}

static void Q1_add(double* matrixC, int ldC, double* m1, double* m4, double* m5, double* m7) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldC / 2; j++) {
		for (int k = 0; k < ldC / 2; k++) {
			matrixC[j * ldC + k] = m1[j * ldC / 2 + k] + m4[j * ldC / 2 + k] - m5[j * ldC / 2 + k] + m7[j * ldC / 2 + k];
		}
	}
}

static void Q2_add(double* matrixC, int ldC, double* m3, double* m5) {
	// creating C12
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldC / 2; j++) {
		for (int k = 0; k < ldC / 2; k++) {
			matrixC[j * ldC + (ldC / 2 + k)] = m3[j * ldC / 2 + k] + m5[j * ldC / 2 + k];
		}
	}
}

static void Q3_add(double* matrixC, int ldC, double* m2, double* m4) {
	// creating C21

#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldC / 2; j++) {
		for (int k = 0; k < ldC / 2; k++) {
			matrixC[(j + ldC / 2) * ldC + k] = m2[j * ldC / 2 + k] + m4[j * ldC / 2 + k];
		}
	}
}

static void Q4_addSubAddAdd(double* matrixC, int ldC, double* m1, double* m2, double* m3, double* m6) {
	// creating C22
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int j = 0; j < ldC / 2; j++) {
		for (int k = 0; k < ldC / 2; k++) {
			matrixC[(j + ldC / 2) * ldC + (ldC / 2 + k)] = m1[j * ldC / 2 + k] - m2[j * ldC / 2 + k] + m3[j * ldC / 2 + k] + m6[j * ldC / 2 + k];
		}
	}
}


/*
* Strassen's Multiplication algorithm using Divide and Conquer technique.
*/
double* strassensMultRec(const double* matrixA, const double* matrixB, double* matrixC, int n, int steps_left, double* work)
{
	double* result = 0;

	if (steps_left == 0)
	{
		//This is the terminating condition for using strassen.
		result = mkl_multiplication(matrixA, matrixB, matrixC, n);
		return result;
	}

	//Divide the matrix
	double* m1 = work;
	double* m2 = &m1[n / 2 * n / 2];
	double* m3 = &m1[2 * n / 2 * n / 2];
	double* m4 = &m1[3 * n / 2 * n / 2];
	double* m5 = &m1[4 * n / 2 * n / 2];
	double* m6 = &m1[5 * n / 2 * n / 2];
	double* m7 = &m1[6 * n / 2 * n / 2];
	double* S = &m1[7 * n / 2 * n / 2];
	double* T = &m1[8 * n / 2 * n / 2];
	double* nextWork = &work[9 * n / 2 * n / 2];

	S1_add(matrixA, S, n);
	T1_add(matrixB, T, n);
	strassensMultRec(S, T, m1, n / 2, steps_left - 1, nextWork);

	S2_add(matrixA, S, n);
	T2_add(matrixB, T, n);
	strassensMultRec(S, T, m2, n / 2, steps_left - 1, nextWork);

	S3_add(matrixA, S, n);
	T3_add(matrixB, T, n);
	strassensMultRec(S, T, m3, n / 2, steps_left - 1, nextWork);

	S4_add(matrixA, S, n);
	T4_add(matrixB, T, n);
	strassensMultRec(S, T, m4, n / 2, steps_left - 1, nextWork);

	S5_add(matrixA, S, n);
	T5_add(matrixB, T, n);
	strassensMultRec(S, T, m5, n / 2, steps_left - 1, nextWork);

	S6_add(matrixA, S, n);
	T6_add(matrixB, T, n);
	strassensMultRec(S, T, m6, n / 2, steps_left - 1, nextWork);

	S7_add(matrixA, S, n);
	T7_add(matrixB, T, n);
	strassensMultRec(S, T, m7, n / 2, steps_left - 1, nextWork);

	// creating C11
	Q1_add(matrixC, n, m1, m4, m5, m7);

	// creating C12
	Q2_add(matrixC, n, m3, m5);

	// creating C21
	Q3_add(matrixC, n, m2, m4);

	// creating C22
	Q4_addSubAddAdd(matrixC, n, m1, m2, m3, m6);

	return matrixC;
}
