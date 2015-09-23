#include "nn.hpp"

using namespace std;
using namespace nn;

template<typename T>
bool equals (const T& t1, const T& t2){
    return abs(t1-t2) <= 0.000001;
}

void vectorPlusTest(){  
    Vector<float> v1(shared_ptr<float>(new float[2]{0.5,0.5}),2);   
    Vector<float> v(shared_ptr<float>(new float[2]{0.2,0.3}),2); 
    v.plus(v1);
    ASSERT(v.size() == 2, "v");
    ASSERT(equals(v.data().get()[0] , 0.7f), "v");
    ASSERT(equals(v.data().get()[1] , 0.8f), "v");      
}

void vectorApplyTest(){
    Vector<float> v(shared_ptr<float>(new float[2]{0.5,0.6}),2);  
    auto inc=[](const float& t) {return t+1;};   
    v.apply(inc);
    ASSERT(v.size() == 2, "v");
    ASSERT(equals(v.data().get()[0] , 1.5f), "v");
    ASSERT(equals(v.data().get()[1] , 1.6f), "v");    
}

void vectorAggregateTest(){
    Vector<float> v(shared_ptr<float>(new float[2]{0.5,0.6}),2);  
    auto sum=[](const float& t1, const float& t2) {return t1+t2;};     
    auto result=v.aggregate(sum,0);   
    ASSERT(equals(result , 1.1f), "result");  
}

void matrixMultiplyTest(){
    Matrix<float> m(shared_ptr<float>(new float[2*3]{1,2,3,4,5,6}),2,3);    
    Vector<float> v(shared_ptr<float>(new float[3]{0.1,0.2,0.3}),3);      
    auto r=m.multiply(v);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , 1.4f), "r[0]");
    ASSERT(equals(r.data().get()[1] , 3.2f), "r[1]"); 
}

void hiddenLayerTest(){
    Matrix<float> W(shared_ptr<float>(new float[2*3]{1,2,3,4,5,6}),2,3);   
    Vector<float> b(shared_ptr<float>(new float[2]{1.5,1.8}),2);
    function<float(const float&)> func(ActivationFunctions<float>::Tanh);  
    HiddenLayer<float> layer(W,b,func);    
    Vector<float> v(shared_ptr<float>(new float[3]{0.1,0.2,0.3}),3);      
    auto r=layer.calc(v);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , tanh(2.9f)), "r[0]");
    ASSERT(equals(r.data().get()[1] , tanh(5.0f)), "r[1]"); 
}

void sequenceInputTest(){
    Matrix<float> E(shared_ptr<float>(new float[3*2]{0,0,0.1,0.2,0.3,0.4}),3,2);
    std::vector<size_t> idSequence = { 0,2 };  
    SequenceInput<float> input(idSequence,1,E,Poolings<Vector<float>>::AVG());
    auto r=input.get();
    ASSERT(r.size() == 2*3, "r");
    ASSERT(equals(r.data().get()[0] , 0.05f), "r");
    ASSERT(equals(r.data().get()[1] , 0.1f), "r");
    ASSERT(equals(r.data().get()[2] , 0.15f), "r");
    ASSERT(equals(r.data().get()[3] , 0.2f), "r");
    ASSERT(equals(r.data().get()[4] , 0.2f), "r");
    ASSERT(equals(r.data().get()[5] , 0.3f), "r");
}

void nonSequenceInputTest(){
    Matrix<float> E(shared_ptr<float>(new float[3*2]{0,0,0.1,0.2,0.3,0.4}),3,2);
    NonSequenceInput<float> input(2,E);
    auto r=input.get();
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , 0.3f), "r");
    ASSERT(equals(r.data().get()[1] , 0.4f), "r");   
}

void inputLayerTest(){
    Matrix<float> E1(shared_ptr<float>(new float[3*2]{0,0,0.1,0.2,0.3,0.4} ),3,2);
    std::vector<size_t> idSequence = { 0,2 };    
    SequenceInput<float> sequenceInput(idSequence,1,E1,Poolings<Vector<float>>::AVG());
    Matrix<float> E2(shared_ptr<float>(new float[3*2]{0,0,0.3,0.8,0.2,0.9} ),3,2);
    NonSequenceInput<float> nonSequenceInput(2,E2);
    vector<reference_wrapper<Input<float>>> inputs={sequenceInput,nonSequenceInput};    
    InputLayer<float> layer;
    auto r =layer.calc(inputs);
    ASSERT(r.size() == 2*3+2, "r");
    ASSERT(equals(r.data().get()[0] , 0.05f), "r");
    ASSERT(equals(r.data().get()[1] , 0.1f), "r");
    ASSERT(equals(r.data().get()[2] , 0.15f), "r");
    ASSERT(equals(r.data().get()[3] , 0.2f), "r");
    ASSERT(equals(r.data().get()[4] , 0.2f), "r");
    ASSERT(equals(r.data().get()[5] , 0.3f), "r");  
    ASSERT(equals(r.data().get()[6] , 0.2f), "r");  
    ASSERT(equals(r.data().get()[7] , 0.9f), "r");  
}

void softmaxLayerTest(){   
    shared_ptr<float> data(new float[2]{0.5,0.5});     
    Vector<float> input(data,2);  
    SoftmaxLayer<float> softmaxLayer;
    Vector<float> result = softmaxLayer.calc(input); 
    ASSERT(result.size() == 2, "result");  
    ASSERT(equals(result.data().get()[0] , 0.5f), "result");
    ASSERT(equals(result.data().get()[1] , 0.5f), "result");
}

void MLPTest(){   
    Matrix<float> E1(shared_ptr<float>(new float[3*2] {0,0,0.1,0.2,0.3,0.4}),3,2);
    Matrix<float> E2(shared_ptr<float>(new float[3*2]{0,0,0.3,0.8,0.2,0.9}),3,2);    
    Embeddings<float> embeddings ({E1,E2});
    shared_ptr<InputLayer<float>> inputLayer (new InputLayer<float>());   
    Matrix<float> W(shared_ptr<float>(new float[2*8] {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6}), 2,8);
    Vector<float> b(shared_ptr<float>(new float[2]{0.1,0.2}),2);
    function<float(const float&)> tanh(ActivationFunctions<float>::Tanh );
   
    std::vector<std::shared_ptr<Layer<float, Vector<float>>>> layers={
        std::shared_ptr<Layer<float, Vector<float>>>(new HiddenLayer<float> (W,b,tanh))
    }; 
    SequenceInput <float> sequenceInput({0,2},1,embeddings.get(0),Poolings<Vector<float>>::AVG());
    NonSequenceInput<float> nonsequenceInput(2,embeddings.get(1));
    vector<reference_wrapper<Input<float>>> inputs={sequenceInput,nonsequenceInput};    
    MLP<float> nn(inputLayer, layers); 
    auto r=nn.calc(inputs);  
    ASSERT(r.size() == 2, "r");      
    ASSERT(equals(r.data().get()[0] ,tanh(0.05*0.1 + 0.1*0.2+ 0.15*0.3 + 0.2*0.4 + 0.2*0.5 + 0.3*0.6 + 0.2*0.7+0.9*0.8 + 0.1)), "r");
    ASSERT(equals(r.data().get()[1] ,tanh(0.05*0.9 + 0.1*1.0+ 0.15*1.1 + 0.2*1.2 + 0.2*1.3 + 0.3*1.4 + 0.2*1.5+0.9*1.6 + 0.2)), "r");    
}

int main( int argc, const char* argv[] )
{
    vectorPlusTest();
    vectorApplyTest();
    vectorAggregateTest();
    matrixMultiplyTest();
    hiddenLayerTest();
    sequenceInputTest();
    nonSequenceInputTest();
    inputLayerTest();
    softmaxLayerTest();
    MLPTest();
}
