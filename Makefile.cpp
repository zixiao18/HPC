MMult1: MMult1.o
	g++ -std=c++11 -fopenmp -O3 -march=native MMult1.cpp && ./fmm1.out

MMult1.o: MMult1.c