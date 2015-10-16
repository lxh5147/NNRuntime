#include "import_theano_model.hpp"
using namespace std;
using namespace nn;
using namespace nn_tools;

template<typename T>
bool equals (Vector<T>& t1, Vector<T>& t2){
    if(t1.size()!=t2.size()){
        return false;
    }
    size_t size=t1.size();
    T* data1=t1.data().get();
    T* data2=t2.data().get();
    for(size_t i=0;i<size;++i){
        if(!equals(data1[i],data2[i])){
            return false;
        }
    }
    return true;
}

//test with the toy model
int main( int argc, const char* argv[] ){
    ASSERT(argc==2,"argc");
    auto path=argv[1];
    auto model=nn_tools::TheanoModel<float>::load(path);
    //sequence: "prefix_1", "prefix_2", "prefix_3", "suffix_1", "suffix_2", "suffix_3", "words" 
    //non sequence:field_ids
    vector<vector<size_t>> idsInputs ={{16,0,3,8,19,10},{157,0,3,90,29,328},{0,0,0,145,0,1683},{10,0,3,11,13,3},{134,0,3,16,27,63},{0,0,0,140,0,100},{611,0,3,175,37,6181}};
    //compare predictions
    auto actual=model->predict(idsInputs);
    Vector<float> expected(make_shared_ptr(new float[12]{5.202937147959219022e-15,7.204790251544562069e-16,4.699654877185821533e-01,2.974174683913588524e-03,7.035437738522887230e-04,3.561305114999413490e-03,5.101346969604492188e-01,1.178610837087035179e-03,4.170435015112161636e-03,6.723115802742540836e-04,3.287334693595767021e-03,3.352134721353650093e-03}),12);
    ASSERT(equals(actual,expected),"prediction");
}
