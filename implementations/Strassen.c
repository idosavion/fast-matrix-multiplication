#include "Strassen.h"


int compare_strassen_and_naive(int dim) {

	int i = 0, j = 0, n = 0;
	n = dim;

	//To handle when n is not power of k we do the padding with zero
	int pow = 1;
	while (pow < n) {
		pow = pow * 2;
	}
	float** matrixA = create(n, pow);
	float** matrixB = create(n, pow);
	n = pow;
	float** standardRes, ** strassenRes;


	clock_t start = clock(), diff;
	float** stdRes = standardMultiplication(matrixA, matrixB, n);
	diff = clock() - start;
	for (int i = 0;i < n;i++)
	{
		free(stdRes[i]);
	}
	free(stdRes);

	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Standard Multiplication took %d msecs\n", msec);
	//printf("\nStandard Multiplication Output:\n");
	//printMatrix(stdRes, n);
	start = clock(), diff;
	//printf("\nStrassen's Multiplication Output:\n");
	float** strassensRes = strassensMultiplication(matrixA, matrixB, n);
	diff = clock() - start;

	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Strassen Multiplication took %d msecs\n", msec);
	//printMatrix(strassensRes, n);

	return 0;
}

/*
* Create the matrix with random floating point numbers betweeen -5 to 5
*/
float** create(int n, int pow) {
	float** array = createZeroMatrix(pow);
	int i, j;
	for (i = 0;i < n; i++) {
		for (j = 0; j < n; j++) {
			array[i][j] = ((((float)rand()) / ((float)RAND_MAX) * 10.0) - 5.0);
		}
	}
	return array;
}

/*
* This function creates the matrix and initialize all elements to zero
*/
float** createZeroMatrix(int n) {
	float** array = (float**)malloc(n * sizeof(float*));
	int i, j;
	for (i = 0;i < n; i++) {
		array[i] = (float*)malloc(n * sizeof(float));
		for (j = 0; j < n; j++) {
			array[i][j] = 0.0;
		}
	}
	return array;
}
/*
* Function to Print Matrix
*/
void printMatrix(float** matrix, int n) {
	int i, j;
	for (i = 0;i < n;i++) {
		for (j = 0;j < n;j++) {
			printf("   %.2f   ", matrix[i][j]);
		}
		printf("\n");
	}
}
/*
* Standard Matrix multiplication with O(n) time complexity.
*/
float** standardMultiplication(float** matrixA, float** matrixB, int n) {
	float** result;
	int i, j, k;

	result = (float**)malloc(n * sizeof(float*));
	for (i = 0;i < n;i++) {
		result[i] = (float*)malloc(n * sizeof(float));
		for (j = 0;j < n;j++) {
			result[i][j] = 0;
			for (k = 0;k < n;k++) {
				result[i][j] = result[i][j] + (matrixA[i][k] * matrixB[k][j]);
			}
		}
	}
	return result;
}

/*
* Wrapper function over strassensMultRec.
*/
float** strassensMultiplication(float** matrixA, float** matrixB, int n) {
	float** result = strassensMultRec(matrixA, matrixB, n, 32);
	return result;
}

/*
* Strassen's Multiplication algorithm using Divide and Conquer technique.
*/
float** strassensMultRec(float** matrixA, float** matrixB, int n, int threshold) {
	float** result;
	if (n > threshold) {
		//Divide the matrix
		result = createZeroMatrix(n);
		float** a11 = divide(matrixA, n, 0, 0);
		float** a12 = divide(matrixA, n, 0, (n / 2));
		float** a21 = divide(matrixA, n, (n / 2), 0);
		float** a22 = divide(matrixA, n, (n / 2), (n / 2));
		float** b11 = divide(matrixB, n, 0, 0);
		float** b12 = divide(matrixB, n, 0, n / 2);
		float** b21 = divide(matrixB, n, n / 2, 0);
		float** b22 = divide(matrixB, n, n / 2, n / 2);

		//Recursive call for Divide and Conquer
		float** m1 = strassensMultRec(addMatrix(a11, a22, n / 2), addMatrix(b11, b22, n / 2), n / 2, threshold);
		float** m2 = strassensMultRec(addMatrix(a21, a22, n / 2), b11, n / 2, threshold);
		float** m3 = strassensMultRec(a11, subMatrix(b12, b22, n / 2), n / 2, threshold);
		float** m4 = strassensMultRec(a22, subMatrix(b21, b11, n / 2), n / 2, threshold);
		float** m5 = strassensMultRec(addMatrix(a11, a12, n / 2), b22, n / 2, threshold);
		float** m6 = strassensMultRec(subMatrix(a21, a11, n / 2), addMatrix(b11, b12, n / 2), n / 2, threshold);
		float** m7 = strassensMultRec(subMatrix(a12, a22, n / 2), addMatrix(b21, b22, n / 2), n / 2, threshold);

		float** c11 = addMatrix(subMatrix(addMatrix(m1, m4, n / 2), m5, n / 2), m7, n / 2);
		float** c12 = addMatrix(m3, m5, n / 2);
		float** c21 = addMatrix(m2, m4, n / 2);
		float** c22 = addMatrix(subMatrix(addMatrix(m1, m3, n / 2), m2, n / 2), m6, n / 2);
		//Compose the matrix
		compose(c11, result, 0, 0, n / 2);
		compose(c12, result, 0, n / 2, n / 2);
		compose(c21, result, n / 2, 0, n / 2);
		compose(c22, result, n / 2, n / 2, n / 2);

		free(c11);
		free(c12);
		free(c21);
		free(c22);
	}
	else {
		//This is the terminating condition for using strassen.
		result = standardMultiplication(matrixA, matrixA, n);
	}
	return result;
}

/*
* This method combines the matrix in the result matrix
*/
void compose(float** matrix, float** result, int row, int col, int n) {
	int i, j, r = row, c = col;
	for (i = 0;i < n;i++) {
		c = col;
		for (j = 0;j < n;j++) {
			result[r][c] = matrix[i][j];
			c++;
		}
		r++;
	}
}

/*
* Sub-divide the matrix according to row and col specified
*/
float** divide(float** matrix, int n, int row, int col) {
	int n_new = n / 2;

	float** array = createZeroMatrix(n_new);
	int i, j, r = row, c = col;
	for (i = 0;i < n_new; i++) {
		c = col;
		for (j = 0; j < n_new; j++) {
			array[i][j] = matrix[r][c];
			c++;
		}
		r++;
	}
	return array;
}

/*
* Add the two input matrix
*/
float** addMatrix(float** matrixA, float** matrixB, int n) {
	float** res = createZeroMatrix(n);
	int i, j;
	for (i = 0;i < n;i++)
		for (j = 0;j < n;j++)
			res[i][j] = matrixA[i][j] + matrixB[i][j];

	return res;
}
/*
* Substract the two matrix
*/
float** subMatrix(float** matrixA, float** matrixB, int n) {
	float** res = createZeroMatrix(n);
	int i, j;
	for (i = 0;i < n;i++)
		for (j = 0;j < n;j++)
			res[i][j] = matrixA[i][j] - matrixB[i][j];

	return res;
}