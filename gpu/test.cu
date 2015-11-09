#include <assert.h>
#include "mvm.h"
#include "mvm.cu"

//Defines helper function to compare two values.
template<typename T, typename U>
bool equals (const T& t1, const U& t2){
    return abs(t1-t2) <= 0.000001;
}

template<typename T>
void testMVM(){
    T A[15]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    T x[3]={0.1,0.2,0.3};
    T y[5];
    CALL_MVM_GPU_FUNC(A,x,y,5,3);
    assert(equals(y[0] , 1.4));
    assert(equals(y[1] , 3.2));
    assert(equals(y[2] , 7*0.1+8*0.2+9*0.3));
    assert(equals(y[3] , 10*0.1+11*0.2+12*0.3));
    assert(equals(y[4] , 13*0.1+14*0.2+15*0.3));
}

int main( int argc, const char* argv[]){
    testMVM<float>();
    testMVM<double>();
}

