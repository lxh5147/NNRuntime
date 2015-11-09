#define DECL_MVM_GPU_FUNC(T) inline void mvm_gpu(const T*, const T* , T* , const size_t, const size_t);

DECL_MVM_GPU_FUNC(float)
DECL_MVM_GPU_FUNC(double)

#define CALL_MVM_GPU_FUNC(A,x,y,row,col) (mvm_gpu(A,x,y,row,col))
