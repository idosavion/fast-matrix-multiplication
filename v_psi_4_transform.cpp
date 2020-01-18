#include "v_psi_4_transform.hpp"

const int FROM_DIM = 14;
const int TO_DIM = 17;


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

static void S2_sum(const double* matrixA, double* S2, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S2[k] = matrixA[( 1 * block_size ) + k];
    }
}

static void S3_sum(const double* matrixA, double* S3, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S3[k] = matrixA[( 9 * block_size ) + k]- matrixA[( 11 * block_size ) + k];
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

static void S5_sum(const double* matrixA, double* S5, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S5[k] = matrixA[( 3 * block_size ) + k];
    }
}

static void S6_sum(const double* matrixA, double* S6, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S6[k] = - matrixA[( 5 * block_size ) + k] + matrixA[( 9 * block_size ) + k];
    }
}

static void S7_sum(const double* matrixA, double* S7, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S7[k] = matrixA[( 4 * block_size ) + k];
    }
}

static void S8_sum(const double* matrixA, double* S8, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S8[k] = matrixA[( 5 * block_size ) + k];
    }
}

static void S9_sum(const double* matrixA, double* S9, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S9[k] = matrixA[( 6 * block_size ) + k];
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

static void S11_sum(const double* matrixA, double* S11, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S11[k] = - matrixA[( 5 * block_size ) + k] + matrixA[( 12 * block_size ) + k];
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

static void S13_sum(const double* matrixA, double* S13, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S13[k] = matrixA[( 9 * block_size ) + k];
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

static void S15_sum(const double* matrixA, double* S15, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S15[k] = matrixA[( 11 * block_size ) + k];
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

static void S17_sum(const double* matrixA, double* S17, int block_size){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int k = 0; k < block_size ; k++){
        S17[k] = matrixA[( 13 * block_size ) + k];
    }
}
void v_psi_4_transform(double* original_a, double* transformed_a, int size, int steps_left, double* work){
    int block_size = size / FROM_DIM;
    int trans_block_size = transformed_block_size(size, steps_left);

    if (steps_left == 0) {
        return;
    }
    else if (steps_left == 1) {
        S1_sum(original_a, transformed_a, block_size);
        S2_sum(original_a, &transformed_a[1*trans_block_size], block_size);
        S3_sum(original_a, &transformed_a[2*trans_block_size], block_size);
        S4_sum(original_a, &transformed_a[3*trans_block_size], block_size);
        S5_sum(original_a, &transformed_a[4*trans_block_size], block_size);
        S6_sum(original_a, &transformed_a[5*trans_block_size], block_size);
        S7_sum(original_a, &transformed_a[6*trans_block_size], block_size);
        S8_sum(original_a, &transformed_a[7*trans_block_size], block_size);
        S9_sum(original_a, &transformed_a[8*trans_block_size], block_size);
        S10_sum(original_a, &transformed_a[9*trans_block_size], block_size);
        S11_sum(original_a, &transformed_a[10*trans_block_size], block_size);
        S12_sum(original_a, &transformed_a[11*trans_block_size], block_size);
        S13_sum(original_a, &transformed_a[12*trans_block_size], block_size);
        S14_sum(original_a, &transformed_a[13*trans_block_size], block_size);
        S15_sum(original_a, &transformed_a[14*trans_block_size], block_size);
        S16_sum(original_a, &transformed_a[15*trans_block_size], block_size);
        S17_sum(original_a, &transformed_a[16*trans_block_size], block_size);
        return;
   }

    double* S = work;
    double* nextWork = &work[block_size];

    S1_sum(original_a, S, block_size);
    v_psi_4_transform(S, transformed_a, block_size, steps_left - 1, nextWork);

    S2_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[1 * trans_block_size], block_size, steps_left - 1, nextWork);

    S3_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[2 * trans_block_size], block_size, steps_left - 1, nextWork);

    S4_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[3 * trans_block_size], block_size, steps_left - 1, nextWork);

    S5_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[4 * trans_block_size], block_size, steps_left - 1, nextWork);

    S6_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[5 * trans_block_size], block_size, steps_left - 1, nextWork);

    S7_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[6 * trans_block_size], block_size, steps_left - 1, nextWork);

    S8_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[7 * trans_block_size], block_size, steps_left - 1, nextWork);

    S9_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[8 * trans_block_size], block_size, steps_left - 1, nextWork);

    S10_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[9 * trans_block_size], block_size, steps_left - 1, nextWork);

    S11_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[10 * trans_block_size], block_size, steps_left - 1, nextWork);

    S12_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[11 * trans_block_size], block_size, steps_left - 1, nextWork);

    S13_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[12 * trans_block_size], block_size, steps_left - 1, nextWork);

    S14_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[13 * trans_block_size], block_size, steps_left - 1, nextWork);

    S15_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[14 * trans_block_size], block_size, steps_left - 1, nextWork);

    S16_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[15 * trans_block_size], block_size, steps_left - 1, nextWork);

    S17_sum(original_a, S, block_size);
    v_psi_4_transform(S, &transformed_a[16 * trans_block_size], block_size, steps_left - 1, nextWork);

}
