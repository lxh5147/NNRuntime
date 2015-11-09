#include "mvm.cuh"
#include <assert.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

template<typename T>
static void mvm_gpu(const T* A, const T* x, T* y, const size_t row, const size_t col){
    assert(A);
    assert(x);
    assert(y);
    assert(row>0);
    assert(col>0);
    T* A_d;
    cudaMalloc((void**)&A_d,sizeof(T)*row*col);
    cudaMemcpy(A_d, A, sizeof(T)*row*col, cudaMemcpyHostToDevice);
    T* x_d;
    cudaMalloc((void**)&x_d,sizeof(T)*row);
    cudaMemcpy(x_d, x, sizeof(T)*row, cudaMemcpyHostToDevice);
    T* y_d;
    cudaMalloc((void**)&y_d,sizeof(T)*row);
    int numberOfBlocks=(row+BLOCK_SIZE-1)/BLOCK_SIZE;
    mvm_kernal<<<numberOfBlocks,BLOCK_SIZE>>> (A_d,x_d,y_d,row,col,BLOCK_SIZE);
    cudaMemcpy(y,y_d,sizeof(T)*row, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(x_d);
    cudaFree(y_d);
}

#define IMPL_MVM_GPU_FUNC(T) extern "C" void mvm_gpu(const ##T* A, const ##T* x, ##T* y, const size_t row, const size_t col) {mvm_gpu<##T>(A,x,y,row,col)}

IMPL_MVM_GPU_FUNC(float);
IMPL_MVM_GPU_FUNC(double);

#endif
