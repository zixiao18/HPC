#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  int n_thread = 3;
  long* partial_sum = (long*) malloc((n_thread-1) * sizeof (long));
  long split = n / n_thread;
  
  #pragma omp parallel for num_threads(n_thread)
  for (long thd=1; thd<n_thread; thd++){
  	  #pragma omp atom write
  	  prefix_sum[split*thd] = 0;
  	  partial_sum[thd] = 0;
	  for (long j = split*thd+1; j < split*(thd+1); j++) {
	  	#pragma omp atom update
	    prefix_sum[j] = prefix_sum[j-1] + A[j-1];
	    partial_sum[thd] += A[j-1];
	  }
	 partial_sum[thd] += A[split*(thd+1)-1]; 
  } 
  // add partial_sum back
  long to_add = 0;
  for (long thd=1; thd<n_thread; thd++){
  	to_add += partial_sum[thd];
  	for (long j=split*thd+1; j<split*(thd+1); j++){
  		prefix_sum[j] += to_add;
	  }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
