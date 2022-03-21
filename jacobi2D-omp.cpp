#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

long double get_res(double *u, double N) {
    double resid = 0.0;
    double h = 1/(N+1); 
    double err = 0.0;
    int n2 = N + 2;
    for (long i = 1; i <= N; i++) { 
        for (long j = 1; j <= N; j++) {
            err = fabs(1-(4*u[i*n2+j]-u[(i-1)*n2+j]-u[i*n2+j-1]-u[(i+1)*n2+j]-u[i*n2+j+1])/h/h);
            resid = std::max(resid,err);
        } 
    }
    return resid;
}

void jacobi(int N) {
	double n = (double) N;
    double h = 1/(n+1);
	int n2 = N+2;
    double* u = (double*) malloc(n2*n2*sizeof(double));
    for (long i = 0; i < n2*n2; i++) u[i] = 0;
    double resid = get_res(u,n);
    double resid0=0.0;
    double* ut = (double*) malloc(n2*n2*sizeof(double));

    // #pragma omp parallel for
    for (long k = 0; k < 5000; k++) {

        #pragma omp parallel 
        {
            #pragma omp for
            for (long i = 0; i < n2*n2; i++) {
                ut[i] = u[i];
            } 

            #pragma omp barrier
            #pragma omp for collapse(2)   
            for (long j=1; j < N+1; j+=1) {
                    for (long i = 1; i < N+1; i+=1) {
                        u[i*n2+j]=(ut[(i-1)*n2+j]+ut[i*n2+j-1]+ut[(i+1)*n2+j]+ut[i*n2+j+1]+1*h*h)/4.0;
                    }
                }
        }

        resid = get_res(u,n);
        if ((resid / resid0) <= 1e-6) {
            printf("The iteration converges\n");
            return;
            }   
        if (k % 500 == 0) {
		printf("After iteration: %ld, the residual is: %f \n", k, resid);
        }
    }
    free(u);
}

int main(int argc, char** argv) {
    Timer t;
    int n = read_option<int>("-n", argc, argv,"100");
    t.tic();
    jacobi(n);
    double time = t.toc();
    
    printf("For matrix of size %d, the time elapsed for 5000 iterations is: %10f\n", n, t.toc());
    return 0;
}
