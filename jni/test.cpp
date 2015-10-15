#include "nn.hpp"
#include "nn_runtime.h"

using namespace std;
using namespace nn;

int main( int argc, const char* argv[] )
{
    //run the test of runtime to generate this sample model
    ASSERT(argc==2,"argc");    
    auto modelFile=argv[1];
    size_t handle=load(modelFile);
    ASSERT(handle,"handle");
    vector<vector<size_t>> idsInputs ={{0,2},{2}};
    auto r=predict(handle,idsInputs);
    auto t1=tanh(0.05*0.1 + 0.1*0.2+ 0.15*0.3 + 0.2*0.4 + 0.2*0.5 + 0.3*0.6 + 0.2*0.7+0.9*0.8 + 0.1);
    auto t2=tanh(0.05*0.9 + 0.1*1.0+ 0.15*1.1 + 0.2*1.2 + 0.2*1.3 + 0.3*1.4 + 0.2*1.5+0.9*1.6 + 0.2);
    auto o1=exp(t1)/(exp(t1)+exp(t2));
    auto o2=exp(t2)/(exp(t1)+exp(t2));
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r[0],o1), "r");
    ASSERT(equals(r[1],o2), "r");
}

