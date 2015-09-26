#include "nn.hpp"
#include "nn_runtime.h"
#include <mutex>

using namespace std;
using namespace nn;

#ifndef TYPE_NN_PARAMETER
#define TYPE_NN_PARAMETER float
#endif

typedef NNModel<PARAMETER_DATA_TYPE,vector<reference_wrapper<Input<TYPE_NN_PARAMETER>>>> TYPE_NNModel;

vector<shared_ptr<TYPE_NNModel>> models;

mutex lock;

const int HANDLE_INVALID=-1;

int load(const char* modelPath){
    ASSERT(modelPath);
    decltype(make_shared_ptr(new TYPE_NNModel())) pModel=nullptr;
    try{
        pModel=make_shared_ptr(new TYPE_NNModel());
        pModel->load(modelPath);
    }
    catch(...){
        cerr<<"Failed to load binary model: "<<modelPath<<endl;
        return HANDLE_INVALID;
    }
    lock.lock();
    try {       
        models.push_back(pModel);
        int handle = models.size()-1;
        lock.unlock(); 
        return handle;
    } catch (const bad_alloc &ex) {
       lock.unlock(); 
       cerr<<ex.what() << ": failed to load binary model: "<<modelPath<<endl;
       return HANDLE_INVALID;
    } 
}

vector<size_t> predict(int modelHandle, const vector<vector<size_t>>& idsInputs){
    ASSERT(modelHandle>0 && modelHandle<models.size(),"modelHandle");
    TYPE_NNModel* pModel=models[modelHandle].get();    
    try {
      return pModel->predict(idsInputs);
    } catch (...) {
        cerr<<"Failed to predict with model handle="<<modelHandle<<endl;
        cerr<<"with id inputs="<<endl;
        for(auto& ids:idsInputs){
            for(auto& id:ids){
                cerr<<id<<" ";
            }
            cerr<<endl;
        }
        return vector<size_t>();
    }
}
