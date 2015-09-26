#include "nn.hpp"
#include "nn_runtime.h"
#include <mutex>

using namespace std;
using namespace nn;

#ifndef TYPE_NN_PARAMETER
#define TYPE_NN_PARAMETER float
#endif

typedef MLPModel<TYPE_NN_PARAMETER> TYPE_MLPModel;

vector<shared_ptr<TYPE_MLPModel>> models;

mutex mx;

const size_t HANDLE_INVALID=0;

size_t load(const char* modelPath){
    ASSERT(modelPath,"modelPath");
    decltype(make_shared_ptr(new TYPE_MLPModel())) pModel=nullptr;
    try{
        pModel=make_shared_ptr(new TYPE_MLPModel());
        pModel->load(modelPath);
    }
    catch(...){
        cerr<<"Failed to load binary model: "<<modelPath<<endl;
        return HANDLE_INVALID;
    }
    mx.lock();
    try {       
        models.push_back(pModel);
        size_t handle = models.size();
        mx.unlock(); 
        return handle;
    } catch (const bad_alloc &ex) {
       mx.unlock(); 
       cerr<<ex.what() << ": failed to load binary model: "<<modelPath<<endl;
       return HANDLE_INVALID;
    } 
}

template<typename T>
vector<double> to_vector(const Vector<T>& input){
    vector<double> result(input.size());
    T* elements=input.data().get();
    for(size_t i=0;i<input.size();++i){
        result.push_back(elements[i]);
    }
    return result;
}

vector<double> predict(size_t modelHandle, const vector<vector<size_t>>& idsInputs){
    ASSERT(modelHandle>0 && modelHandle<models.size(),"modelHandle");
    TYPE_MLPModel* pModel=models[modelHandle-1].get();    
    try {
      return to_vector(pModel->predict(idsInputs));
    } catch (...) {
        cerr<<"Failed to predict with model handle="<<modelHandle<<endl;
        cerr<<"with id inputs="<<endl;
        for(auto& ids:idsInputs){
            for(auto& id:ids){
                cerr<<id<<" ";
            }
            cerr<<endl;
        }
        return vector<double>();
    }
}
