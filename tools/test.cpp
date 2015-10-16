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

template<typename T>
bool equals (Matrix<T>& m1, Matrix<T>& m2){
    if(m1.row()!=m2.row()){
        return false;
    }
    if(m1.col()!=m2.col()){
        return false;
    }
    size_t size=m1.row()*m1.col();
    T* data1=m1.data().get();
    T* data2=m2.data().get();
    for(size_t i=0;i<size;++i){
        if(!equals(data1[i],data2[i])){
            return false;
        }
    }
    return true;
}

void transpose_test(){
    auto w=newMatrix(new float[2*3]{1,2,3,4,5,6},2,3);
    auto actual=transpose(*w);
    auto expected=newMatrix(new float[3*2]{1,4,2,5,3,6},3,2);
    ASSERT(equals(*actual,*expected),"transpose");
}

void json_util_get_value_test(){
    ASSERT(JsonUtil::getJsonValue("{\"num_hidden\": 2, \"conv\": 32}","num_hidden")=="2","num_hidden");
    ASSERT(JsonUtil::getJsonValue("{\"num_hidden\": 2, \"conv\": 32}","conv")=="32","conv");
    ASSERT(JsonUtil::getJsonValue("{\"name\": \"zeen\", \"conv\": 32}","name")=="zeen","name");
}

void json_util_get_string_values_test(){
    vector<string> expectedValues1={"v1"};
    ASSERT(JsonUtil::getJsonStringValues("{\"values1\": [\"v1\"], \"values2\": [\"prefix_1\", \"prefix_2\"]","values1")==expectedValues1,"values1");
    vector<string> expectedValues2={"prefix_1","prefix_2"};
    ASSERT(JsonUtil::getJsonStringValues("{\"values1\": [\"v1\", \"v2\" ], \"values2\": [\"prefix_1\", \"prefix_2\"]","values2")==expectedValues2,"values2");
}
//dump  for debug purpose
template<typename T>
void dump(const Vector<T>& vector){
    cout<<"size="<<vector.size()<<endl;
    T* data=vector.data().get();
    for(size_t i=0;i<vector.size();++i){
        cout<<data[i]<<" ";
    }
    cout<<endl;
}

template<typename T>
void dump(const Matrix<T>& matrix){
    size_t row=matrix.row();
    size_t col=matrix.col();
    cout<<"row="<<row<<","<<"col="<<col<<endl;
    T* data=matrix.data().get();
    cout.setf(ios::fixed,ios::floatfield);
    cout.precision(5);
    for(size_t i=0;i<row;++i){
        for(size_t j=0;j<col;++j){
           cout<<data[i*col+j]<<" ";
        }
        cout<<endl;
    }
}

void integration_test(const string& path){
    auto model=nn_tools::TheanoModel<float>::load(path);
    //sequence: "prefix_1", "prefix_2", "prefix_3", "suffix_1", "suffix_2", "suffix_3", "words"
    //non sequence:field_ids
    vector<vector<size_t>> idsInputs ={{16,0,3,8,19,10},{157,0,3,90,29,328},{0,0,0,145,0,1683},{10,0,3,11,13,3},{134,0,3,16,27,63},{0,0,0,140,0,100},{611,0,3,175,37,6181},{3}};
    //compare predictions
    auto actual=model->predict(idsInputs);
    Vector<float> expected(make_shared_ptr(new float[12]{5.202937147959219022e-15,7.204790251544562069e-16,4.699654877185821533e-01,2.974174683913588524e-03,7.035437738522887230e-04,3.561305114999413490e-03,5.101346969604492188e-01,1.178610837087035179e-03,4.170435015112161636e-03,6.723115802742540836e-04,3.287334693595767021e-03,3.352134721353650093e-03}),12);
    cout<<"actual"<<endl;
    dump(actual);
    cout<<"expected:"<<endl;
    dump(expected);
    ASSERT(equals(actual,expected),"prediction");
}

int main( int argc, const char* argv[] ){
    transpose_test();
    json_util_get_value_test();
    json_util_get_string_values_test();
    ASSERT(argc==2,"argc");
    //integration test with toy model
    integration_test(argv[1]);
}
