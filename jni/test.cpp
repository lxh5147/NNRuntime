#include "nn.hpp"
#include "nn.test.hpp"
#include "nn_runtime.h"

using namespace std;
using namespace nn;
using namespace nn::test;


template<class T>
void generateTestModel(const string& modelFile){
        //describe model in memory
    DECL_MATRIX_PTR(T,E1,3,2,{0,0,0.1,0.2,0.3,0.4})
    DECL_MATRIX_PTR(T,E2,3,2,{0,0,0.3,0.8,0.2,0.9})
    vector<shared_ptr<Matrix<T> > > embeddings;
    embeddings.push_back(E1);
    embeddings.push_back(E2);
    shared_ptr<Embedding<T> > pEmbedding1=EmbeddingWithRawValues<T>::create(*embeddings[0]);
    shared_ptr<Embedding<T> > pEmbedding2=EmbeddingWithRawValues<T>::create(*embeddings[1]);
    vector<shared_ptr<InputInfo<T> > > inputsInfo;
    inputsInfo.push_back(newInputInfo(pEmbedding1,1,Poolings<Vector<T> >::AVG));
    inputsInfo.push_back(newInputInfo(pEmbedding2));
    //two hidden layers
    DECL_MATRIX_PTR(T,W1,2,8,{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6})
    DECL_MATRIX_PTR(T,W2,2,2,{0.2,0.3,0.1,0.5})
    vector<shared_ptr<Matrix<T> > > weights;
    weights.push_back(W1);
    weights.push_back(W2);
    DECL_VECTOR_PTR(T,b1,2,{0.1,0.2})
    DECL_VECTOR_PTR(T,b2,2,{0.3,0.4})
    vector<shared_ptr<Vector<T> > > biasVectors;
    biasVectors.push_back(b1);
    biasVectors.push_back(b2);
    vector<size_t> activationFunctionIds;
    activationFunctionIds.push_back(ActivationFunctions<T>::TANH);
    activationFunctionIds.push_back(ActivationFunctions<T>::TANH);
    //save and load model
    MLPModelFactory::save<T>(modelFile,inputsInfo,weights,biasVectors,activationFunctionIds);
}

template<class T, bool quantizeEmbedding>
void loadAndPredictTest(string modelFile){
    size_t handle=load(modelFile.c_str(),quantizeEmbedding,true);
    ASSERT(handle,"handle");
    vector<vector<size_t> > idsInputs;
    DECL_STD_VECTOR(size_t,ids1,2,{0,2})
    DECL_STD_VECTOR(size_t,ids2,1,{2})
    idsInputs.push_back(ids1);
    idsInputs.push_back(ids2);
    double oh11=tanh(0.05*0.1 + 0.1*0.2+ 0.15*0.3 + 0.2*0.4 + 0.2*0.5 + 0.3*0.6 + 0.2*0.7+0.9*0.8 + 0.1);
    double oh12=tanh(0.05*0.9 + 0.1*1.0+ 0.15*1.1 + 0.2*1.2 + 0.2*1.3 + 0.3*1.4 + 0.2*1.5+0.9*1.6 + 0.2);
    double t1=tanh(oh11*0.2 + oh12*0.3+0.3);
    double t2=tanh(oh11*0.1 + oh12*0.5+0.4);
    //soft max
    double o1=exp(t1)/(exp(t1)+exp(t2));
    double o2=exp(t2)/(exp(t1)+exp(t2));
    vector<double> r=predict(handle,idsInputs);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r[0],o1), "r");
    ASSERT(equals(r[1],o2), "r");
}

int main( int argc, const char* argv[] )
{
    string modelFile="model.test.bin";
    generateTestModel<TYPE_NN_PARAMETER>(modelFile);
    loadAndPredictTest<TYPE_NN_PARAMETER,false>(modelFile);
}

