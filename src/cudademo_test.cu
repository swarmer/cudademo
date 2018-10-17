#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>


const int VECTOR_SIZE = 1024;


__global__ void vector_add(int a[], int b[], int out[], size_t size) {
    const size_t i = threadIdx.x;

    if (i < size) {
        out[i] = a[i] + b[i];
    }
}


int main(int argc, char *argv[]) {
    int *a, *b, *out;
    cudaMallocManaged(&a, VECTOR_SIZE * sizeof(int));
    cudaMallocManaged(&b, VECTOR_SIZE * sizeof(int));
    cudaMallocManaged(&out, VECTOR_SIZE * sizeof(int));

    for (size_t i = 0; i < VECTOR_SIZE; ++i) {
        a[i] = i;
        b[i] = i + 1;
        out[i] = 0;
    }

    vector_add<<<1, VECTOR_SIZE>>>(a, b, out, VECTOR_SIZE);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < VECTOR_SIZE; ++i) {
        printf("out[%zd]: %d\n", i, out[i]);
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(out);

    return 0;
}
