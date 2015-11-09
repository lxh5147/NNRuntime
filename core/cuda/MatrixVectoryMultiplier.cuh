#ifndef _MVM_KERNAL_
#define _MVM_KERNAL_

#include<cassert>
#include<cuda.h>
#include<cuda_runtime.h>

namespace nn{
    //y=A*x using shared memory, refer to https://sites.google.com/site/bsb3166/cuda_tutorial
    template<typename T>
    __global__ void mvm_kernal(const T *A, const T* x, T* y, const size_t row, const size_t col, const size_t blockSize)
    {
        //each thread calculates the dot product of one row; 
        __shared__ T Xds[blockSize];
        int bx=blockIdx.x; 
        int tx=threadIdx.x; 
        size_t curRow=bx*blockSize+tx;
        T value=0;
        for (size_t m=0; m<(col-1)/blockSize+1;++m){
            if(m*blockSize+tx<col){
                Xds[tx]=x[m*blockSize+tx]; 
            }else{
                Xds[tx]=0;
            }
            __syncthreads();
            for (size_t k=0; k<blockSize;k++){
                 if(curRow<row && m*blockSize+k<col){
                    value+=A[m*blockSize+curRow*col+k]*Xds[k];
                }
            }
            __syncthreads();
        }
        if(curRow<row)
            y[curRow]=value; 
        __syncthreads();
    }

    #define BLOCK_SIZE 4

    template <typename T>
    class MatrixVectoryMultiplierCuda{
        public:
            static inline void multiply(const T* A, const T* x, T* y, const size_t row, const size_t col){
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
                int numberOfWordsBlocks=(row+BLOCK_SIZE-1)/BLOCK_SIZE;
                mvm_kernal<<<numberOfWordsBlocks,BLOCK_SIZE>>> (A_d,x_d,y_d,row,col,BLOCK_SIZE);
                cudaMemcpy(y,y_d,sizeof(T)*row, cudaMemcpyDeviceToHost);
                cudaFree(A_d);
                cudaFree(x_d);
                cudaFree(y_d);
           }
    };

    typedef MatrixVectoryMultiplierCuda<float> MatrixVectoryMultiplierCudaFloat;
    typedef MatrixVectoryMultiplierCuda<double> MatrixVectoryMultiplierCudaDouble;
}
#endif
