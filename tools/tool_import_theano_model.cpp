#include "import_theano_model.hpp"

#ifndef TYPE_NN_PARAMETER
#define TYPE_NN_PARAMETER float
#endif

typedef nn_tools::TheanoModel<TYPE_NN_PARAMETER> TYPE_TheanoModel;

int main( int argc, const char* argv[] )
{
    ASSERT(argc>=3,"argc");
    //usage: inputTheanoModelFolder outputRuntimeModelFile [includeOutputSoftmaxLaye:true|false]
    auto inputTheanoModelFolder=argv[1];
    auto outputRuntimeModelFile=argv[2];
    auto includeOutputSoftmaxLayer=true;
    if(argc>=4){
        if(strcmp("false",argv[3])==0){
            includeOutputSoftmaxLayer=false;
        }
    }
    auto runtimeModel=TYPE_TheanoModel::load(inputTheanoModelFolder,includeOutputSoftmaxLayer);
    runtimeModel->save(outputRuntimeModelFile);
}
