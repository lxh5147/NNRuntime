#include "nn.hpp"
#include "nn_runtime.h"

using namespace std;
using namespace common;
using namespace nn;

#ifndef TYPE_Embedding
#define TYPE_Embedding EmbeddingWith16BitsQuantizedValues
#endif

#ifndef TYPE_MatrixVectoryMultiplier
#define TYPE_MatrixVectoryMultiplier MatrixVectoryMultiplier
#endif

typedef MLPModel<TYPE_NN_PARAMETER,TYPE_MatrixVectoryMultiplier> TYPE_MLPModel;

//Loaded models.
static Cache<TYPE_MLPModel,size_t> models;

const size_t HANDLE_INVALID=0;

size_t load(const char* modelPath,bool quantizeEmbedding,bool normalizeOutputWithSoftmax){
    ASSERT(modelPath,"modelPath");
    shared_ptr<TYPE_MLPModel> pModel;
    try{
        #ifdef DEBUG
        cout<<"DEBUG\tload model with modelPath:"<<modelPath<<",quantizeEmbedding:"<<quantizeEmbedding<<",normalizeOutputWithSoftmax:"<<normalizeOutputWithSoftmax<<endl;
        #endif
        #ifdef PERF
        clock_t wctstart=clock();
        #endif
        pModel=quantizeEmbedding?MLPModelFactory::load<TYPE_NN_PARAMETER,TYPE_Embedding,TYPE_MatrixVectoryMultiplier>(modelPath,normalizeOutputWithSoftmax):MLPModelFactory::load<TYPE_NN_PARAMETER,TYPE_Embedding,TYPE_MatrixVectoryMultiplier>(modelPath,normalizeOutputWithSoftmax);
        #ifdef PERF
        clock_t wctduration = (clock()-wctstart);
        cout << "PERF\tload finished in " << microseconds(wctduration) << " micro seconds (Wall Clock)" << endl;
        #endif
        size_t handle=models.put(pModel);
        #ifdef DEBUG
        cout<<"DEBUG\tloaded model handle:"<<handle<<endl;
        #endif
        return handle;
    }
    catch(...){
        cerr<<"Failed to load binary model "<<modelPath<<endl;
        return HANDLE_INVALID;
    } 
}

//Helper function that converts Vector to std::vector.
template<typename T>
vector<double> to_vector(const Vector<T>& input){
    vector<double> result;
    result.reserve(input.size());
    T* elements=input.data().get();
    for(size_t i=0;i<input.size();++i){
        result.push_back(elements[i]);
    }
    return result;
}

//Predicts the probability of each category.
vector<double> predict(size_t modelHandle, const vector<vector<size_t> >& idsInputs){
    #ifdef DEBUG
    cout<<"DEBUG\tpredict with modelHandle:"<<modelHandle<<",idsInputs:";
    BEGIN_STD_VECTOR_FOR(idsInputs,ids,vector<size_t>)
        cout<<"[ ";
        BEGIN_STD_VECTOR_FOR(ids,id,size_t)
            cout<<id<<" ";
        END_STD_VECTOR_FOR
        cout<<"]";
    END_STD_VECTOR_FOR
    cout<<endl;
    #endif
    shared_ptr<TYPE_MLPModel> model=models.get(modelHandle);
    ASSERT(model,"model");
    try {
        #ifdef PERF
        clock_t wctstart=clock();
        #endif
        vector<double> prediction=to_vector(model->predict(idsInputs));
        #ifdef PERF
        clock_t wctduration = (clock()-wctstart);
        cout << "PERF\tpredict finished in " << microseconds(wctduration) << " micro seconds (Wall Clock)" << endl;
        #endif
        #ifdef DEBUG
        cout<<"DEBUG\tpredict:";
        BEGIN_STD_VECTOR_FOR(prediction,prob,double)
            cout<<prob<<" ";
        END_STD_VECTOR_FOR
        cout<<endl;
        #endif
        return prediction;
    } catch (...) {
        cerr<<"Failed to predict with: model handle="<<modelHandle<<endl;
        cerr<<"Id inputs="<<endl;
        BEGIN_STD_VECTOR_FOR(idsInputs,ids,vector<size_t>)
            cerr<<"[ ";
            BEGIN_STD_VECTOR_FOR(ids,id,size_t)
                cerr<<id<<" ";
            END_STD_VECTOR_FOR
            cerr<<"]";
        END_STD_VECTOR_FOR
        cerr<<endl;
        return vector<double>();
    }
}
