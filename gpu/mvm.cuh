#ifndef _MVM_GPU_KERNAL_CUH_
#define _MVM_GPU_KERNAL_CUH_

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

//y=A*x using shared memory, refer to https://sites.google.com/site/bsb3166/cuda_tutorial
template<typename T>
__global__ void mvm_gpu_kernal(const T *A, const T* x, T* y, const size_t row, const size_t col)
{
    //each thread calculates the dot product of one row; 
    __shared__ T Xds[BLOCK_SIZE];
    int bx=blockIdx.x; 
    int tx=threadIdx.x; 
    size_t curRow=bx*BLOCK_SIZE+tx;
    T value=0;
    for (size_t m=0; m<(col-1)/BLOCK_SIZE+1;++m){
        if(m*BLOCK_SIZE+tx<col){
            Xds[tx]=x[m*BLOCK_SIZE+tx]; 
        }else{
            Xds[tx]=0;
        }
        __syncthreads();
        for (size_t k=0; k<BLOCK_SIZE;k++){
             if(curRow<row && m*BLOCK_SIZE+k<col){
                value+=A[m*BLOCK_SIZE+curRow*col+k]*Xds[k];
            }
        }
        __syncthreads();
    }
    if(curRow<row){
        y[curRow]=value; 
    }
    __syncthreads();
}

#endif
