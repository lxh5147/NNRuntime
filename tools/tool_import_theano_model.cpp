#include "import_theano_model.hpp"

int main( int argc, const char* argv[] )
{
    ASSERT(argc>=3,"argc");
    //usage: inputTheanoModelFolder outputRuntimeModelFile [includeOutputSoftmaxLaye:true|false]
    const char* inputTheanoModelFolder=argv[1];
    const char* outputRuntimeModelFile=argv[2];
    bool includeOutputSoftmaxLayer=true;
    if(argc>=4){
        std::string option=argv[3];
        if(option=="false"){
            includeOutputSoftmaxLayer=false;
        }
    }
    nn::tools::import_theano_model<TYPE_NN_PARAMETER>(inputTheanoModelFolder,outputRuntimeModelFile,includeOutputSoftmaxLayer);
}
