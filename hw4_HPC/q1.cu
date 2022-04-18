// We made modifications based on gpu03.cu shown in class
// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <math.h>

#define N 1024 // row size 
#define M 512 // column size
#define BLOCK_SIZE 256

void vec_mul (double* A, double* B, double *C){
    //#pragma omp parallel for collapse(2)
    for (long i = 0; i < N; i++) {
        double inner_prod = 0;
        for (long j = 0; j < M; j++) {
            //#pragma omp atomic update 
            inner_prod += A[i*M+j] * B[j];
        }
        C[i] = inner_prod;
    }
}

__global__ 
void vec_mul_kernel(double* A, double* B, double* C, double* temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double value = 0;
        for (long i = 0; i < M; i++) {
            value += A[idx*M+i] * B[i];
        }
        C[idx] = value;
    }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main(int argc, char** argcv) {
    long col_size = M;
    long row_size = N;

    double* A = (double*) malloc(col_size*row_size * sizeof(double));
    double* B = (double*) malloc(col_size * sizeof(double));
    double* C = (double*) malloc(row_size * sizeof(double));
    double* C_ref = (double*) malloc(row_size * sizeof(double));

    for (long i = 0; i < col_size*row_size; i++) A[i] = drand48();
    for (long i = 0; i < col_size; i++) B[i] = drand48();
    for (long i = 0; i < row_size; i++) C[i] = 0.0;
    for (long i = 0; i < row_size; i++) C_ref[i] = 0.0;

    double tt = omp_get_wtime();
    vec_mul(A,B,C_ref);
    printf("CPU %f s\n", omp_get_wtime()-tt);
    double cpu_bandwidth = 3*M*N*sizeof(double)/(omp_get_wtime()-tt)/1e9;
    printf("CPU bandwidth = %f GB/s\n", cpu_bandwidth);

    double *A_d, *B_d, *C_d, *temp;
    cudaMalloc(&A_d, col_size*row_size*sizeof(double));
    Check_CUDA_Error("malloc x failed");
    cudaMalloc(&B_d, col_size*sizeof(double));
    cudaMalloc(&C_d, row_size*sizeof(double));
    cudaMalloc(&temp, col_size*row_size*sizeof(double));

    tt = omp_get_wtime();
    cudaMemcpy(A_d, A, col_size*row_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, col_size*sizeof(double), cudaMemcpyHostToDevice);
    double ttinner = omp_get_wtime();
    vec_mul_kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(A_d,B_d,C_d,temp);
    cudaDeviceSynchronize();
    ttinner = omp_get_wtime() - ttinner;
    cudaMemcpy(C, C_d, row_size*sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

    double bandwidth = 3*M*N*sizeof(double)/(omp_get_wtime()-tt)/1e9;
    printf("GPU bandwidth = %f GB/s\n", bandwidth);

    double err = 0;
    for (long i = 0; i < N; i++) err += fabs(C[i]-C_ref[i]);
    printf("Error = %f\n", err);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(temp);

    free(A);
    free(B);
    free(C);
    free(C_ref);

    return 0;
}
