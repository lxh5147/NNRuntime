#include "import_theano_model.hpp"
#include "nn.test.hpp"

using namespace std;
using namespace nn;
using namespace nn::test;
using namespace nn::tools;
using namespace boost;

template<typename T>
void transpose_test(){
    DECL_MATRIX(T,w,2,3,{1,2,3,4,5,6})
    DECL_MATRIX(T,expected,3,2,{1,4,2,5,3,6})
    Matrix<T> actual=*transpose(w);
    ASSERT(equals(actual,expected),"transpose");
}

void json_util_get_value_test(){
    ASSERT(JsonUtil::getJsonValue("{\"num_hidden\": 2, \"conv\": 32}","num_hidden")=="2","num_hidden");
    ASSERT(JsonUtil::getJsonValue("{\"num_hidden\": 2, \"conv\": 32}","conv")=="32","conv");
    ASSERT(JsonUtil::getJsonValue("{\"name\": \"zeen\", \"conv\": 32}","name")=="zeen","name");
}

void json_util_get_string_values_test(){
    DECL_STD_VECTOR(string,expectedValues1,1,{"v1"})
    ASSERT(JsonUtil::getJsonStringValues("{\"values1\": [\"v1\"], \"values2\": [\"prefix_1\", \"prefix_2\"]","values1")==expectedValues1,"values1");
    DECL_STD_VECTOR(string,expectedValues2,2,{"prefix_1","prefix_2"})
    ASSERT(JsonUtil::getJsonStringValues("{\"values1\": [\"v1\", \"v2\" ], \"values2\": [\"prefix_1\", \"prefix_2\"]","values2")==expectedValues2,"values2");
}


template<class T,template<class> class E,template<class> class MVM>
void integration_test(const string& toyModelPath){
    //classifier without quantization
    shared_ptr<MLPModel<T,MVM> > model=MLPModelFactory::load<T,E,MVM>(toyModelPath);
    //sequence: "prefix_1", "prefix_2", "prefix_3", "suffix_1", "suffix_2", "suffix_3", "words"
    //non sequence:field_ids
    DECL_STD_VECTOR(size_t,ids1,6,{16,0,3,8,19,10})
    DECL_STD_VECTOR(size_t,ids2,6,{157,0,3,90,29,328})
    DECL_STD_VECTOR(size_t,ids3,6,{0,0,0,145,0,1683})
    DECL_STD_VECTOR(size_t,ids4,6,{10,0,3,11,13,3})
    DECL_STD_VECTOR(size_t,ids5,6,{134,0,3,16,27,63})
    DECL_STD_VECTOR(size_t,ids6,6,{0,0,0,140,0,100})
    DECL_STD_VECTOR(size_t,ids7,6,{611,0,3,175,37,6181})
    DECL_STD_VECTOR(size_t,ids8,1,{3})
    vector<vector<size_t> > idsInputs;
    INIT_STD_VECTOR(vector<size_t>,idsInputs,8,{ids1,ids2,ids3,ids4,ids5,ids6,ids7,ids8})
    //compare predictions
    Vector<T> actual=model->predict(idsInputs);
    DECL_VECTOR(T,expected,12,{3.805000324557826150e-15,5.585751888525787145e-16,4.636411666870117188e-01,2.968471962958574295e-03,6.844294839538633823e-04,3.390415105968713760e-03,5.174831748008728027e-01,1.115844235755503178e-03,4.151279106736183167e-03,5.496459780260920525e-04,2.982420381158590317e-03,3.033172106370329857e-03})
    ASSERT(equals(actual,expected),"prediction");
}

int main( int argc, const char* argv[] ){
    transpose_test<float>();
    transpose_test<double>();
    json_util_get_value_test();
    json_util_get_string_values_test();
    ASSERT(argc==2,"argc");
    //integration test with toy model
    integration_test<float,EmbeddingWithRawValues,MatrixVectoryMultiplier>(argv[1]);
}
