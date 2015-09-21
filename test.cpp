#include "nn.hpp"

using namespace std;
using namespace nn;

template<typename T>
bool equals (const T& t1, const T& t2){
    return abs(t1-t2) <= 0.000001;
}

void vectorPlusTest(){
    float* t = new float[2];
    t[0]=0.5;
    t[1]=0.5;  
    Vector<float> v1(shared_ptr<float>(t),2);     
    t = new float[2];
    t[0]=0.2;
    t[1]=0.3; 
    Vector<float> v(shared_ptr<float>(t),2); 
    v.plus(v1);
    ASSERT(v.size() == 2, "v");
    ASSERT(equals(v.data().get()[0] , 0.7f), "v");
    ASSERT(equals(v.data().get()[1] , 0.8f), "v");      
}

void vectorApplyTest(){
    auto t = new float[2];
    t[0]=0.5;
    t[1]=0.6;  
    Vector<float> v(shared_ptr<float>(t),2);  
    auto inc=[](const float& t) {return t+1;};   
    v.apply(inc);
    ASSERT(v.size() == 2, "v");
    ASSERT(equals(v.data().get()[0] , 1.5f), "v");
    ASSERT(equals(v.data().get()[1] , 1.6f), "v");    
}

void vectorAggregateTest(){
    auto t = new float[2];
    t[0]=0.5;
    t[1]=0.6;  
    Vector<float> v(shared_ptr<float>(t),2);  
    auto sum=[](const float& t1, const float& t2) {return t1+t2;};     
    auto result=v.aggregate(sum,0);   
    ASSERT(equals(result , 1.1f), "result");  
}

void matrixMultiplyTest(){
    auto t = new float[2*3];
    //first row: 1 2 3 
    t[0]=1;
    t[1]=2;
    t[2]=3;
    //second row: 4 5 6
    t[3]=4;
    t[4]=5;
    t[5]=6;
    Matrix<float> m(shared_ptr<float>(t),2,3);
    
    t = new float[3];
    t[0]=0.1;
    t[1]=0.2; 
    t[2]=0.3;      
    Vector<float> v(shared_ptr<float>(t),3);  
    
    auto r=m.multiply(v);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , 1.4f), "r[0]");
    ASSERT(equals(r.data().get()[1] , 3.2f), "r[1]"); 
}

void hiddenLayerTest(){
    auto t = new float[2*3];
    //first row: 1 2 3 
    t[0]=1;
    t[1]=2;
    t[2]=3;
    //second row: 4 5 6
    t[3]=4;
    t[4]=5;
    t[5]=6;
    Matrix<float> W(shared_ptr<float>(t),2,3);
    
    t=new float[2];
    t[0]=1.5;
    t[1]=1.8;
    Vector<float> b(shared_ptr<float>(t),2);
    function<float(const float&)> func=ActivationFunctions<float>::Tanh;  
    HiddenLayer<float> layer(W,b,func);     
    t = new float[3];
    t[0]=0.1;
    t[1]=0.2; 
    t[2]=0.3;      
    Vector<float> v(shared_ptr<float>(t),3);  
    
    auto r=layer.calc(v);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , tanh(2.9f)), "r[0]");
    ASSERT(equals(r.data().get()[1] , tanh(5.0f)), "r[1]"); 
}

void softmaxLayerTest(){
    float* t = new float[2];
    t[0]=0.5;
    t[1]=0.5;
    shared_ptr<float> data(t);     
    Vector<float> input(data,2);  
    //return input; 
    SoftmaxLayer<float> softmaxLayer;
    Vector<float> result = softmaxLayer.calc(input); 
    ASSERT(result.size() == 2, "result");  
    ASSERT(equals(result.data().get()[0] , 0.5f), "result");
    ASSERT(equals(result.data().get()[1] , 0.5f), "result");
}

int main( int argc, const char* argv[] )
{
    vectorPlusTest();
    vectorApplyTest();
    vectorAggregateTest();
    matrixMultiplyTest();
    hiddenLayerTest();
    softmaxLayerTest();
}
