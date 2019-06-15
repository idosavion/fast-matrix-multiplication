#include "mkl.h"
#include <time.h>
#include <stdio.h>
#include "mkl_strassen.h"
#define LOOP_COUNT 10

void mult_a11b11(int n, double* matrixA, double* matrixB, double* matrixC)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            n / 2, n / 2, n / 2, 1, matrixA, n, matrixB, n, 0, matrixC, n);
}

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
	double* matrixA = mkl_malloc(pow * pow * sizeof(double), 64);
	double* matrixB = mkl_malloc(pow * pow * sizeof(double), 64);

	double* matrixC_naive = mkl_malloc(pow * pow * sizeof(double), 64);
	double* matrixC_strassen = mkl_malloc(pow * pow * sizeof(double), 64);
	for (int i = 0; i < n * n; i+=2)
	{
		matrixA[i] = 0;
		matrixA[i+1] = i;
		
		matrixB[i] = 0;
		matrixB[i + 1] = i + 1;
		matrixC_naive[i] = 0;
		matrixC_strassen[i] = 0;
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


	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		matrixC_naive = strassensMultRec(matrixA, matrixB, matrixC_naive, n,1);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Strassen Multiplication (1 step) took %.2f msecs\n", msec);

	
	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		matrixC_naive = strassensMultRec(matrixA, matrixB, matrixC_naive, n,2);
	}
	// printMatrix(matrixC_naive, pow);
	diff = clock() - start;
	msec = (double)diff * 1000 / (CLOCKS_PER_SEC * LOOP_COUNT);
	printf("Strassen Multiplication (2 step) took %.2f msecs\n", msec);

	start = clock(), diff;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		matrixC_naive = strassensMultRec(matrixA, matrixB, matrixC_naive, n,3);
	}
	// printMatrix(matrixC_naive, pow);
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
	mkl_free(matrixC_strassen);

	return 0;
}

/*
* Function to Print Matrix
*/
void printMatrix(double* matrix, int n)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			printf("   %.4f   ", matrix[i * n + j]);
		}
		printf("\n");
	}
	printf("\n");
}

/*
* Standard Matrix multiplication with O(n) time complexity.
*/
double* mkl_multiplication(const double* matrixA, const double* matrixB, double* matrixC, int n)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            n, n, n, 1, matrixA, n, matrixB, n, 0, matrixC, n);
	return matrixC;
}

/*
* Wrapper function over strassensMultRec.
*/
double* strassensMultiplication(const double* matrixA, const double* matrixB, double* matrixC, int n, int steps_left)
{
	double* result = strassensMultRec(matrixA, matrixB, matrixC, n, steps_left);
	return result;
}

void create_m1(const double* matrixA, const double* matrixB, int n, int steps_left, double* m1)
{
	double* add_A11_A22 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, matrixA, n, 1, &matrixA[n * n / 2 + n / 2], n, add_A11_A22, n / 2);
	//printf("A11+A22:\n:");
	//printMatrix(add_A11_A22, n / 2);
	double* add_B11_B22 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, matrixB, n, 1, &matrixB[n * n / 2 + n / 2], n, add_B11_B22, n / 2);
	//printf("B11+B22:\n:");
	//printMatrix(add_B11_B22, n / 2);
	strassensMultRec(add_A11_A22, add_B11_B22, m1, n / 2, steps_left - 1);
	//printf("\nMatrix M1:\n");
	//printMatrix(m1, n / 2);
	mkl_free(add_A11_A22);
	mkl_free(add_B11_B22);
}

void create_m2(const double* matrixA, const double* matrixB, int n, int steps_left, double* m2)
{
	double* add_A21_A22 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, &matrixA[n * n / 2], n, 1, &matrixA[n * n / 2 + n / 2], n, add_A21_A22,
	             n / 2);
	//printf("A21+A22:\n:");
	//printMatrix(add_A21_A22, n / 2);
	double* b11 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, matrixB, n, 0, &matrixB[n * n / 2 + n / 2], n, b11, n / 2);
	//printf("b11:\n:");
	//printMatrix(b11, n / 2);
	strassensMultRec(add_A21_A22, b11, m2, n / 2, steps_left - 1);
	//printf("\nMatrix M2:\n");
	//printMatrix(m2, n / 2);
	mkl_free(add_A21_A22);
	mkl_free(b11);
}

void create_m3(const double* matrixA, const double* matrixB, int n, int steps_left, double* m3)
{
	double* sub_b12_b22 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, &matrixB[n / 2], n, -1, &matrixB[n * n / 2 + n / 2], n, sub_b12_b22,
	             n / 2);
	//printf("B12-B22:\n:");
	//printMatrix(sub_b12_b22, n / 2);
	double* a11 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, matrixA, n, 0, &matrixB[n * n / 2 + n / 2], n, a11, n / 2);
	//printf("A11:\n:");
	//printMatrix(a11, n / 2);
	strassensMultRec(a11,sub_b12_b22, m3, n / 2, steps_left - 1);
	//printf("\nMatrix M3:\n");
	//printMatrix(m3, n / 2);
	mkl_free(sub_b12_b22);
	mkl_free(a11);
}

void create_m4(const double* matrixA, const double* matrixB, int n, int steps_left, double* m4)
{
	double* sub_b21_b11 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, &matrixB[n * n / 2], n, -1, matrixB, n, sub_b21_b11,
	             n / 2);
	//printf("B21-B11:\n:");
	//printMatrix(sub_b21_b11, n / 2);
	double* a22 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, &matrixA[n* n /2 + n/2], n, 0, &matrixB[n * n / 2 + n / 2], n, a22, n / 2);
	//printf("A22:\n:");
	//printMatrix(a22, n / 2);
	strassensMultRec(a22,sub_b21_b11, m4, n / 2, steps_left - 1);
	//printf("\nMatrix M4:\n");
	//printMatrix(m4, n / 2);
	mkl_free(sub_b21_b11);
	mkl_free(a22);
}

void create_m5(const double* matrixA, const double* matrixB, int n, int steps_left, double* m5)
{
	double* add_a11_a12 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, matrixA, n, 1, &matrixA[n / 2], n, add_a11_a12,
	             n / 2);
	//printf("A11+A12:\n:");
	//printMatrix(add_a11_a12, n / 2);
	double* b22 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 0, matrixB, n, 1, &matrixB[n * n / 2 + n / 2], n, b22, n / 2);
	//printf("B22:\n:");
	//printMatrix(b22, n / 2);
	strassensMultRec(add_a11_a12, b22, m5, n / 2, steps_left - 1);
	//printf("\nMatrix M5:\n");
	//printMatrix(m5, n / 2);
	mkl_free(add_a11_a12);
	mkl_free(b22);
}

void create_m6(const double* matrixA, const double* matrixB, int n, int steps_left, double* m6)
{
	double* sub_a21_a11 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, &matrixA[n * n / 2], n, -1, matrixA, n, sub_a21_a11,
	             n / 2);
	//printf("A21-A11:\n:");
	//printMatrix(sub_a21_a11, n / 2);
	double* add_b11_b12 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, matrixB, n, 1, &matrixB[n / 2], n, add_b11_b12, n / 2);
	//printf("B11 + B12:\n:");
	//printMatrix(add_b11_b12, n / 2);
	strassensMultRec(sub_a21_a11, add_b11_b12, m6, n / 2, steps_left - 1);
	//printf("\nMatrix M6:\n");
	//printMatrix(m6, n / 2);
	mkl_free(sub_a21_a11);
	mkl_free(add_b11_b12);
}


void create_m7(const double* matrixA, const double* matrixB, int n, int steps_left, double* m7)
{
	double* sub_a12_a22 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, &matrixA[n / 2], n, -1, &matrixA[n * n / 2 + n / 2], n, sub_a12_a22,
	             n / 2);
	//printf("A12-A22:\n:");
	//printMatrix(sub_a12_a22, n / 2);
	double* add_b21_b22 = create_zero_mat(n / 2);
	mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, &matrixB[n * n / 2], n, 1, &matrixB[n * n / 2 + n / 2], n, add_b21_b22,
	             n / 2);
	//printf("B21 + B22:\n:");
	//printMatrix(add_b21_b22, n / 2);
	strassensMultRec(sub_a12_a22, add_b21_b22, m7, n / 2, steps_left - 1);
	//printf("\nMatrix M7:\n");
	//printMatrix(m7, n / 2);
	mkl_free(sub_a12_a22);
	mkl_free(add_b21_b22);
}


/*
* Strassen's Multiplication algorithm using Divide and Conquer technique.
*/
double* strassensMultRec(const double* matrixA, const double* matrixB, double* matrixC, int n, int steps_left)
{
	double* result = 0;
	if (steps_left > 0)
	{
		//Divide the matrix
		double* m1 = create_zero_mat(n / 2);
		double* m2 = create_zero_mat(n / 2);
		double* m3 = create_zero_mat(n / 2);
		double* m4 = create_zero_mat(n / 2);
		double* m5 = create_zero_mat(n / 2);
		double* m6 = create_zero_mat(n / 2);
		double* m7 = create_zero_mat(n / 2);

		create_m1(matrixA, matrixB, n, steps_left, m1);
		create_m2(matrixA, matrixB, n, steps_left, m2);
		create_m3(matrixA, matrixB, n, steps_left, m3);
		create_m4(matrixA, matrixB, n, steps_left, m4);
		create_m5(matrixA, matrixB, n, steps_left, m5);
		create_m6(matrixA, matrixB, n, steps_left, m6);
		create_m7(matrixA, matrixB, n, steps_left, m7);

		// printf("M1:\n:");
		// printMatrix(m1, n/2);
		// printf("M2:\n:");
		// printMatrix(m2, n / 2);
		// printf("M3:\n:");
		// printMatrix(m3, n/2);
		// printf("M4:\n:");
		// printMatrix(m4, n/2);
		// printf("M5:\n:");
		// printMatrix(m5, n/2);
		// printf("M6:\n:");
		// printMatrix(m6, n/2);
		// printf("M7:\n:");
		// printMatrix(m7, n/2);

		// creating C11
		//printMatrix(matrixC, n);

		mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, m1, n / 2, 1, m4, n / 2,matrixC, n);
		// printf("C11 = M1 + M4:\n");
		//printMatrix(matrixC, n);

		mkl_domatadd('R', 'N', 'N', n / 2, n / 2, -1, m5, n / 2, 1, matrixC, n,matrixC, n);
		//printf("C11 = C11 - M5:\n");
		//printMatrix(matrixC, n);

		mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, m7, n / 2, 1, matrixC, n,matrixC, n);
		//printf("C11 = C11 + M7:\n");

		//printMatrix(matrixC, n);
		
		// creating C21
		mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, m3, n / 2, 1, m5, n / 2, &matrixC[n / 2], n);
		//printMatrix(matrixC, n);
		// creating C12
		mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, m2, n / 2, 1, m4, n / 2, &matrixC[n * n / 2], n);
		//printMatrix(matrixC, n);
		// creating C22
		mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, m1, n / 2, -1, m2, n / 2, &matrixC[n * n /2 + n/2], n);
		//printf("C22 = M1 - M2:\n");
		//printMatrix(matrixC, n);

		mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, m3, n / 2, 1, matrixC, n, &matrixC[n * n / 2 + n / 2], n);
		//printf("C22 = C22 + M3:\n");
		//printMatrix(matrixC, n);

		mkl_domatadd('R', 'N', 'N', n / 2, n / 2, 1, m6, n / 2, 1, matrixC, n, &matrixC[n * n / 2 + n / 2], n);
		//printf("C22 = C22 + M6:\n");
		//printMatrix(matrixC, n);

		mkl_free(m1);
		mkl_free(m2);
		mkl_free(m3);
		mkl_free(m4);
		mkl_free(m5);
		mkl_free(m6);
		mkl_free(m7);
		
	}
	else
	{
		//This is the terminating condition for using strassen.
		result = mkl_multiplication(matrixA, matrixB, matrixC, n);
	}
	return matrixC;
}

/*
* This method combines the matrix in the result matrix
*/
double* create_zero_mat(int dim)
{
	double* mat = mkl_malloc(dim * dim * sizeof(double), 64);
	for (int i = 0; i < dim * dim; i++)
	{
		mat[i] = 0;
	}
	return mat;
}
