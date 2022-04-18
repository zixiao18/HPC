#include <algorithm>
#include <stdio.h>
//#include <utils.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

long double calculate_resid(double *u, long* f, double N) {
    double resid = 0.0;
    double err = 0.0;
    double h = 1/(N+1); 
    int np2 = (int) N + 2;
    for (long i = 1; i <= N; i++) { 
        for (long j = 1; j <= N; j++) {
            err = fabs(f[i*np2+j]-(4*u[i*np2+j]-u[(i-1)*np2+j]-u[i*np2+j-1]-u[(i+1)*np2+j]-u[i*np2+j+1])/h/h);
            resid = std::max(resid,err);
        } 
    }
    return resid;
}

void jacobi_cpu(double* unew, double* uold, long* f, int N, double h) {
    int np2 = N+2;
    double residual = calculate_resid(unew,f,N);

    for (long k = 0; k < 5000; k++) {
        #ifdef _OPENMP
            #pragma omp parallel
        #endif
        {
            #ifdef _OPENMP
               #pragma omp for
            #endif
            for (long i = 0; i < np2*np2; i++) {
                uold[i] = unew[i];
            } 
            #ifdef _OPENMP
                #pragma omp barrier
                #pragma omp for collapse(2)
            #endif  
            for (long j=1; j < N+1; j+=1) {
                    for (long i = 1; i < N+1; i+=1) {
                        unew[i*np2+j]=(uold[(i-1)*np2+j]+uold[i*np2+j-1]+uold[(i+1)*np2+j]+uold[i*np2+j+1]+f[i*np2+j]*h*h)/4.0;
                    }
                }
        }
        double resid_temp = calculate_resid(unew,f,N);
        
    }
}

__global__
void jacobi_gpu(double* unew, double* uold, long* f, const int N, const double h){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>0 && i<N-1 && j >0 && j<N-1) {
        unew[i*N+j] = (uold[(i-1)*N+j] + uold[(i+1)*N+j] + uold[i*N+j+1] + uold[i*N+j-1] +f[i*N+j]*h*h)/4.0;
    }
}

int main(int argc, char** argv) {
    int N = 192;
    int np2 = (int) N + 2;
    double* unew = (double*) malloc(np2*np2*sizeof(double));
    double* uold = (double*) malloc(np2*np2*sizeof(double));
    double* utemp = (double*) malloc(np2*np2*sizeof(double));
    long* f = (long*) malloc(np2*np2*sizeof(long));
    double n = (double) N;
    double h = 1/(n+1);
   

    // initialization
    for (long i = 0; i < np2*np2; i++) unew[i] = 0;
    for (long i = 0; i < np2*np2; i++) f[i] = 1.0;
    double residual = calculate_resid(unew,f,N);

    double *unew_d, *uold_d;
    long* f_d;
    cudaMalloc(&unew_d, np2*np2*sizeof(double));
    cudaMalloc(&uold_d, np2*np2*sizeof(double));
    cudaMalloc(&f_d, np2*np2*sizeof(long));

    double tt = omp_get_wtime();
    cudaMemcpy(unew_d, unew, np2*np2*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(f_d, f, np2*np2*sizeof(long), cudaMemcpyHostToDevice);
    double copy_t = omp_get_wtime()-tt;

    tt = omp_get_wtime();
    jacobi_cpu(unew,uold,f,N,h);
    printf("CPU %f s\n", omp_get_wtime()-tt);

    dim3 block(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 grid(np2/BLOCK_SIZE_X, np2/BLOCK_SIZE_Y);

    tt = omp_get_wtime();
    for (int k=0; k<5000; k+=2) {
        jacobi_gpu<<<grid,block>>>(uold_d,unew_d,f_d,N+2,h);
        cudaDeviceSynchronize();
        jacobi_gpu<<<grid,block>>>(unew_d,uold_d, f_d,N+2,h);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(utemp, unew_d, np2*np2*sizeof(double), cudaMemcpyDeviceToHost);
    tt = omp_get_wtime()-tt;
    printf("GPU %f s, %f s\n", tt+copy_t, tt);

    double error = 0;
    for (int i=0; i<N; i++) error += (utemp[i] - unew[i]);

    printf("Error: %f \n", error);
   
    free(unew);
    free(uold);
    free(utemp);
    free(f);
    cudaFree(unew_d);
    cudaFree(uold_d);
    cudaFree(f_d);
    return 0;

}
