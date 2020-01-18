#include "bs_recursive.hpp"

const int FROM_DIM = 20;
const int TO_DIM = 23;

void mkl_multiplication(const double* matrixA, const double* matrixB, double* matrixC, int n) {
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		n, n, n, 1, matrixA, n, matrixB, n, 0, matrixC, n);
}

static inline int transformed_block_size(int block_size, int steps_left) {
    int base_block_size = block_size / (pow(FROM_DIM, steps_left));
    int power = pow(TO_DIM, steps_left-1);

    return power * base_block_size;
}


static void S1_sum(const double* matrixA, double* S1, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S1[k] = matrixA[k];
    }
}

static void T1_sum(const double* matrixB, double* T1, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T1[k] = - matrixB[( 18 * block_size ) + k];
    }
}

static void Q1_sum(double* matrixC, const double* Q1, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[k] += Q1[k];
        }

}

static void S2_sum(const double* matrixA, double* S2, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S2[k] = - matrixA[( 6 * block_size ) + k];
    }
}

static void T2_sum(const double* matrixB, double* T2, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T2[k] = matrixB[( 11 * block_size ) + k];
    }
}

static void Q2_sum(double* matrixC, const double* Q2, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[k] += Q2[k];
        }

}

static void S3_sum(const double* matrixA, double* S3, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S3[k] = matrixA[( 1 * block_size ) + k];
    }
}

static void T3_sum(const double* matrixB, double* T3, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T3[k] = matrixB[k];
    }
}

static void Q3_sum(double* matrixC, const double* Q3, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(1 * block_size) + k] += Q3[k];
        }

}

static void S4_sum(const double* matrixA, double* S4, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S4[k] = matrixA[( 2 * block_size ) + k];
    }
}

static void T4_sum(const double* matrixB, double* T4, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T4[k] = matrixB[( 12 * block_size ) + k];
    }
}

static void Q4_sum(double* matrixC, const double* Q4, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(2 * block_size) + k] += Q4[k];
        }

}

static void S5_sum(const double* matrixA, double* S5, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S5[k] = matrixA[( 3 * block_size ) + k];
    }
}

static void T5_sum(const double* matrixB, double* T5, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T5[k] = matrixB[( 1 * block_size ) + k];
    }
}

static void Q5_sum(double* matrixC, const double* Q5, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(3 * block_size) + k] += Q5[k];
        }

}

static void S6_sum(const double* matrixA, double* S6, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S6[k] = matrixA[( 4 * block_size ) + k];
    }
}

static void T6_sum(const double* matrixB, double* T6, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T6[k] = matrixB[( 2 * block_size ) + k];
    }
}

static void Q6_sum(double* matrixC, const double* Q6, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(4 * block_size) + k] += Q6[k];
        }

}

static void S7_sum(const double* matrixA, double* S7, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S7[k] = matrixA[( 5 * block_size ) + k];
    }
}

static void T7_sum(const double* matrixB, double* T7, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T7[k] = matrixB[( 3 * block_size ) + k];
    }
}

static void Q7_sum(double* matrixC, const double* Q7, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(5 * block_size) + k] += Q7[k];
        }

}

static void S8_sum(const double* matrixA, double* S8, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S8[k] = matrixA[( 6 * block_size ) + k];
    }
}

static void T8_sum(const double* matrixB, double* T8, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T8[k] = matrixB[( 4 * block_size ) + k];
    }
}

static void Q8_sum(double* matrixC, const double* Q8, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(6 * block_size) + k] += Q8[k];
        }

}

static void S9_sum(const double* matrixA, double* S9, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S9[k] = - matrixA[( 12 * block_size ) + k];
    }
}

static void T9_sum(const double* matrixB, double* T9, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T9[k] = matrixB[( 5 * block_size ) + k];
    }
}

static void Q9_sum(double* matrixC, const double* Q9, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(7 * block_size) + k] += Q9[k];
        }

}

static void S10_sum(const double* matrixA, double* S10, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S10[k] = matrixA[( 7 * block_size ) + k];
    }
}

static void T10_sum(const double* matrixB, double* T10, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T10[k] = matrixB[( 6 * block_size ) + k];
    }
}

static void Q10_sum(double* matrixC, const double* Q10, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(8 * block_size) + k] += Q10[k];
        }

}

static void S11_sum(const double* matrixA, double* S11, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S11[k] = - matrixA[( 19 * block_size ) + k];
    }
}

static void T11_sum(const double* matrixB, double* T11, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T11[k] = matrixB[( 7 * block_size ) + k];
    }
}

static void Q11_sum(double* matrixC, const double* Q11, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(9 * block_size) + k] += Q11[k];
        }

}

static void S12_sum(const double* matrixA, double* S12, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S12[k] = matrixA[( 8 * block_size ) + k];
    }
}

static void T12_sum(const double* matrixB, double* T12, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T12[k] = matrixB[( 8 * block_size ) + k];
    }
}

static void Q12_sum(double* matrixC, const double* Q12, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(10 * block_size) + k] += Q12[k];
        }

}

static void S13_sum(const double* matrixA, double* S13, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S13[k] = matrixA[( 9 * block_size ) + k];
    }
}

static void T13_sum(const double* matrixB, double* T13, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T13[k] = matrixB[( 9 * block_size ) + k];
    }
}

static void Q13_sum(double* matrixC, const double* Q13, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(11 * block_size) + k] += Q13[k];
        }

}

static void S14_sum(const double* matrixA, double* S14, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S14[k] = matrixA[( 10 * block_size ) + k];
    }
}

static void T14_sum(const double* matrixB, double* T14, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T14[k] = matrixB[( 10 * block_size ) + k];
    }
}

static void Q14_sum(double* matrixC, const double* Q14, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(12 * block_size) + k] += Q14[k];
        }

}

static void S15_sum(const double* matrixA, double* S15, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S15[k] = matrixA[( 11 * block_size ) + k];
    }
}

static void T15_sum(const double* matrixB, double* T15, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T15[k] = matrixB[( 11 * block_size ) + k];
    }
}

static void Q15_sum(double* matrixC, const double* Q15, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(18 * block_size) + k] += Q15[k];
        }

}

static void S16_sum(const double* matrixA, double* S16, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S16[k] = matrixA[( 12 * block_size ) + k];
    }
}

static void T16_sum(const double* matrixB, double* T16, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T16[k] = matrixB[( 12 * block_size ) + k];
    }
}

static void Q16_sum(double* matrixC, const double* Q16, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(13 * block_size) + k] += Q16[k];
        }

}

static void S17_sum(const double* matrixA, double* S17, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S17[k] = matrixA[( 13 * block_size ) + k];
    }
}

static void T17_sum(const double* matrixB, double* T17, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T17[k] = matrixB[( 13 * block_size ) + k];
    }
}

static void Q17_sum(double* matrixC, const double* Q17, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(19 * block_size) + k] += Q17[k];
        }

}

static void S18_sum(const double* matrixA, double* S18, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S18[k] = matrixA[( 14 * block_size ) + k];
    }
}

static void T18_sum(const double* matrixB, double* T18, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T18[k] = matrixB[( 14 * block_size ) + k];
    }
}

static void Q18_sum(double* matrixC, const double* Q18, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(14 * block_size) + k] += Q18[k];
        }

}

static void S19_sum(const double* matrixA, double* S19, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S19[k] = matrixA[( 15 * block_size ) + k];
    }
}

static void T19_sum(const double* matrixB, double* T19, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T19[k] = matrixB[( 15 * block_size ) + k];
    }
}

static void Q19_sum(double* matrixC, const double* Q19, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(15 * block_size) + k] += Q19[k];
        }

}

static void S20_sum(const double* matrixA, double* S20, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S20[k] = matrixA[( 16 * block_size ) + k];
    }
}

static void T20_sum(const double* matrixB, double* T20, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T20[k] = matrixB[( 16 * block_size ) + k];
    }
}

static void Q20_sum(double* matrixC, const double* Q20, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(16 * block_size) + k] += Q20[k];
        }

}

static void S21_sum(const double* matrixA, double* S21, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S21[k] = matrixA[( 17 * block_size ) + k];
    }
}

static void T21_sum(const double* matrixB, double* T21, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T21[k] = matrixB[( 17 * block_size ) + k];
    }
}

static void Q21_sum(double* matrixC, const double* Q21, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(17 * block_size) + k] += Q21[k];
        }

}

static void S22_sum(const double* matrixA, double* S22, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S22[k] = matrixA[( 18 * block_size ) + k];
    }
}

static void T22_sum(const double* matrixB, double* T22, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T22[k] = matrixB[( 18 * block_size ) + k];
    }
}

static void Q22_sum(double* matrixC, const double* Q22, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(18 * block_size) + k] += Q22[k];
        }

}

static void S23_sum(const double* matrixA, double* S23, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S23[k] = matrixA[( 19 * block_size ) + k];
    }
}

static void T23_sum(const double* matrixB, double* T23, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        T23[k] = matrixB[( 19 * block_size ) + k];
    }
}

static void Q23_sum(double* matrixC, const double* Q23, int block_size){
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int k = 0; k < block_size ; k++){
                matrixC[(19 * block_size) + k] += Q23[k];
        }

}

static void reset_M(double* M, int block_size) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int k = 0; k < block_size; k++) {
		M[k] = 0;
	}
}

void recursive_decomposed_bs(double* matrixA, double* matrixB, double* matrixC, int size, int steps_left, double* work){
    int block_size = size / FROM_DIM;
    if (steps_left == 0) {
        mkl_multiplication(matrixA, matrixB, matrixC, sqrt(size));        return;
    }

    double* S = work;
    double* T = &work[block_size];
    double* M = &work[2 * block_size];
    double* nextWork = &work[3 * block_size];

    S1_sum(matrixA, S, block_size);
    T1_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q1_sum(matrixC, M, block_size);

    S2_sum(matrixA, S, block_size);
    T2_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q2_sum(matrixC, M, block_size);

    S3_sum(matrixA, S, block_size);
    T3_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q3_sum(matrixC, M, block_size);

    S4_sum(matrixA, S, block_size);
    T4_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q4_sum(matrixC, M, block_size);

    S5_sum(matrixA, S, block_size);
    T5_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q5_sum(matrixC, M, block_size);

    S6_sum(matrixA, S, block_size);
    T6_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q6_sum(matrixC, M, block_size);

    S7_sum(matrixA, S, block_size);
    T7_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q7_sum(matrixC, M, block_size);

    S8_sum(matrixA, S, block_size);
    T8_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q8_sum(matrixC, M, block_size);

    S9_sum(matrixA, S, block_size);
    T9_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q9_sum(matrixC, M, block_size);

    S10_sum(matrixA, S, block_size);
    T10_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q10_sum(matrixC, M, block_size);

    S11_sum(matrixA, S, block_size);
    T11_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q11_sum(matrixC, M, block_size);

    S12_sum(matrixA, S, block_size);
    T12_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q12_sum(matrixC, M, block_size);

    S13_sum(matrixA, S, block_size);
    T13_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q13_sum(matrixC, M, block_size);

    S14_sum(matrixA, S, block_size);
    T14_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q14_sum(matrixC, M, block_size);

    S15_sum(matrixA, S, block_size);
    T15_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q15_sum(matrixC, M, block_size);

    S16_sum(matrixA, S, block_size);
    T16_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q16_sum(matrixC, M, block_size);

    S17_sum(matrixA, S, block_size);
    T17_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q17_sum(matrixC, M, block_size);

    S18_sum(matrixA, S, block_size);
    T18_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q18_sum(matrixC, M, block_size);

    S19_sum(matrixA, S, block_size);
    T19_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q19_sum(matrixC, M, block_size);

    S20_sum(matrixA, S, block_size);
    T20_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q20_sum(matrixC, M, block_size);

    S21_sum(matrixA, S, block_size);
    T21_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q21_sum(matrixC, M, block_size);

    S22_sum(matrixA, S, block_size);
    T22_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q22_sum(matrixC, M, block_size);

    S23_sum(matrixA, S, block_size);
    T23_sum(matrixB, T, block_size);
	reset_M(M, block_size);
    recursive_decomposed_bs(S, T, M, block_size, steps_left - 1, nextWork);
    Q23_sum(matrixC, M, block_size);

}
