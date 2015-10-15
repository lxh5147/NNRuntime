#include "import_theano_model.hpp"

#ifndef TYPE_NN_PARAMETER
#define TYPE_NN_PARAMETER float
#endif

typedef nn_tools::TheanoModel<TYPE_NN_PARAMETER> TYPE_TheanoModel;

int main( int argc, const char* argv[] )
{
    ASSERT(argc==3,"argc");
    //usage: inputTheanoModelFolder outputRuntimeModelFile
    auto inputTheanoModelFolder=argv[1];
    auto outputRuntimeModelFile=argv[2];
    auto runtimeModel=TYPE_TheanoModel::load(inputTheanoModelFolder);
    runtimeModel->save(outputRuntimeModelFile);
}
