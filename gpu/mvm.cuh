#ifndef _MVM_GPU_KERNAL_CUH_
#define _MVM_GPU_KERNAL_CUH_

//y=A*x using shared memory, refer to https://sites.google.com/site/bsb3166/cuda_tutorial
template<typename T>
__global__ void mvm_gpu_kernal(const T *A, const T* x, T* y, const size_t row, const size_t col, const size_t blockSize)
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

#endif
