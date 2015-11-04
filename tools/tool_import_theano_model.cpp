#include <string>
#include "import_theano_model.hpp"

#ifndef TYPE_NN_PARAMETER
#define TYPE_NN_PARAMETER float
#endif

int main( int argc, const char* argv[] )
{
    ASSERT(argc>=3,"argc");
    //usage: inputTheanoModelFolder outputRuntimeModelFile [includeOutputSoftmaxLaye:true|false]
    auto inputTheanoModelFolder=argv[1];
    auto outputRuntimeModelFile=argv[2];
    auto includeOutputSoftmaxLayer=true;
    if(argc>=4){
        std::string option=argv[3];
        if(option=="false"){
            includeOutputSoftmaxLayer=false;
        }
    }
    nn_tools::import_theano_model<TYPE_NN_PARAMETER>(inputTheanoModelFolder,outputRuntimeModelFile,includeOutputSoftmaxLayer);
}
