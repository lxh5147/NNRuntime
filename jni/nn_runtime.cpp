#include "nn.hpp"
#include "nn_runtime.h"
#include <mutex>

using namespace std;
using namespace nn;

#ifndef TYPE_NN_PARAMETER
#define TYPE_NN_PARAMETER float
#endif

typedef MLPModel<TYPE_NN_PARAMETER> TYPE_MLPModel;

//Loaded models.
vector<shared_ptr<TYPE_MLPModel>> models;

//Lock associated with the models
mutex modelsLock;

const size_t HANDLE_INVALID=0;

size_t load(const char* modelPath,bool normalizeOutputWithSoftmax){
    ASSERT(modelPath,"modelPath");
    decltype(make_shared_ptr(new TYPE_MLPModel())) pModel=nullptr;
    try{
        pModel=make_shared_ptr(new TYPE_MLPModel(normalizeOutputWithSoftmax));
        pModel->load(modelPath);
    }
    catch(...){
        cerr<<"Failed to load binary model "<<modelPath<<endl;
        return HANDLE_INVALID;
    }
    try {
        modelsLock.lock();
        models.push_back(pModel);
        size_t handle = models.size();
        modelsLock.unlock(); 
        return handle;
    } catch (const bad_alloc &ex) {
       modelsLock.unlock(); 
       cerr<<ex.what() << ": failed to load binary model "<<modelPath<<endl;
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
vector<double> predict(size_t modelHandle, const vector<vector<size_t>>& idsInputs){
    ASSERT(modelHandle>0 && modelHandle<=models.size(),"modelHandle");
    TYPE_MLPModel* pModel=models[modelHandle-1].get();    
    try {
      return to_vector(pModel->predict(idsInputs));
    } catch (...) {
        cerr<<"Failed to predict with: model handle="<<modelHandle<<endl;
        cerr<<"Id inputs="<<endl;
        for(auto& ids:idsInputs){
            for(auto& id:ids){
                cerr<<id<<" ";
            }
            cerr<<endl;
        }
        return vector<double>();
    }
}
