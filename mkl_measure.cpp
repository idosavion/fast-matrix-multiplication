//
//  mkl_measure.cpp
//  matrix_multiplication
//
//  Created by Ido Savion on 14/10/2019.
//  Copyright © 2019 Ido Savion. All rights reserved.
//
#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <time.h>

static void measure_mkl(int dim) {
        double *A, *B, *C;
        int m, n, k, i, j;
        double alpha, beta;

        m = dim, k = dim, n = dim;
        alpha = 1.0; beta = 0.0;

        A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
        B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
        C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
        if (A == NULL || B == NULL || C == NULL) {
          printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
          mkl_free(A);
          mkl_free(B);
          mkl_free(C);
        }
        for (i = 0; i < (m*k); i++) {
            A[i] = (double)(i+1);
        }

        for (i = 0; i < (k*n); i++) {
            B[i] = (double)(-i-1);
        }

        for (i = 0; i < (m*n); i++) {
            C[i] = 0.0;
        }
        clock_t t;
        t = clock();
    for(int i=0; i < 10;i++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);


    }

        t = clock() - t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
          printf("mkl multiplication took %f seconds to execute \n", time_taken);

    }

