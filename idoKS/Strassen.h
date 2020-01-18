#ifndef STRASSEN_H
#define STRASSEN_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float** create(int, int);
float** createZeroMatrix(int);
float** strassensMultiplication(float**, float**, int);
float** standardMultiplication(float**, float**, int);
float** strassensMultRec(float**, float**, int n, int threshold);
float** divide(float** matrixA, int n, int row, int col);
void printMatrix(float**, int);
float** addMatrix(float**, float**, int);
float** subMatrix(float**, float**, int);
void compose(float**, float**, int, int, int);

#endif //STRASSEN_H
