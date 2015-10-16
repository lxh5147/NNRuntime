#include "nn.hpp"

using namespace std;
using namespace nn;

void vectorPlusTest(){
    Vector<float> v1(make_shared_ptr(new float[2]{0.5,0.5}),2);
    Vector<float> v(make_shared_ptr(new float[2]{0.2,0.3}),2);
    v.plus(v1);
    ASSERT(v.size() == 2, "v");
    ASSERT(equals(v.data().get()[0] , 0.7f), "v");
    ASSERT(equals(v.data().get()[1] , 0.8f), "v");
}

void vectorDivideTest(){
    Vector<float> v(make_shared_ptr(new float[2]{0.2,0.4}),2);
    v.divide(2);
    ASSERT(v.size() == 2, "v");
    ASSERT(equals(v.data().get()[0] , 0.1), "v");
    ASSERT(equals(v.data().get()[1] , 0.2), "v");
}

void vectorMaxTest(){
    Vector<float> v1(make_shared_ptr(new float[2]{0.3,0.4}),2);
    Vector<float> v(make_shared_ptr(new float[2]{0.5,0.1}),2);
    v.max(v1);
    ASSERT(v.size() == 2, "v");
    ASSERT(equals(v.data().get()[0] , 0.5), "v");
    ASSERT(equals(v.data().get()[1] , 0.4), "v");
}

void vectorApplyTest(){
    Vector<float> v(make_shared_ptr(new float[2]{0.5,0.6}),2);
    auto inc=[](const float& t) {return t+1;};
    v.apply(inc);
    ASSERT(v.size() == 2, "v");
    ASSERT(equals(v.data().get()[0] , 1.5f), "v");
    ASSERT(equals(v.data().get()[1] , 1.6f), "v");
}

void vectorAggregateTest(){
    Vector<float> v(make_shared_ptr(new float[2]{0.5,0.6}),2);
    auto sum=[](const float& t1, const float& t2) {return t1+t2;};
    auto result=v.aggregate(sum,0);
    ASSERT(equals(result , 1.1f), "result");
}

void matrixMultiplyTest(){
    Matrix<float> m(make_shared_ptr(new float[2*3]{1,2,3,4,5,6}),2,3);
    Vector<float> v(make_shared_ptr(new float[3]{0.1,0.2,0.3}),3);
    auto r=m.multiply(v);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , 1.4f), "r[0]");
    ASSERT(equals(r.data().get()[1] , 3.2f), "r[1]");
}

void hiddenLayerTest(){
    Matrix<float> W(make_shared_ptr(new float[2*3]{1,2,3,4,5,6}),2,3);
    Vector<float> b(make_shared_ptr(new float[2]{1.5,1.8}),2);
    HiddenLayer<float> layer(W,b,ActivationFunctions<float>::get(0));
    Vector<float> v(make_shared_ptr(new float[3]{0.1,0.2,0.3}),3);
    auto r=layer.calc(v);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , tanh(2.9f)), "r[0]");
    ASSERT(equals(r.data().get()[1] , tanh(5.0f)), "r[1]");
}

void sequenceInputTest(){
    Matrix<float> E(make_shared_ptr(new float[3*2]{0,0,0.1,0.2,0.3,0.4}),3,2);
    std::vector<size_t> idSequence = { 0,2 };
    SequenceInput<float> input(idSequence,1,E,Poolings<Vector<float>>::get(0));
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
    Matrix<float> E(make_shared_ptr(new float[3*2]{0,0,0.1,0.2,0.3,0.4}),3,2);
    NonSequenceInput<float> input(2,E);
    auto r=input.get();
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , 0.3f), "r");
    ASSERT(equals(r.data().get()[1] , 0.4f), "r");
}

void inputLayerTest(){
    Matrix<float> E1(make_shared_ptr(new float[3*2]{0,0,0.1,0.2,0.3,0.4} ),3,2);
    std::vector<size_t> idSequence = { 0,2 };
    SequenceInput<float> sequenceInput(idSequence,1,E1,Poolings<Vector<float>>::get(0));
    Matrix<float> E2(make_shared_ptr(new float[3*2]{0,0,0.3,0.8,0.2,0.9} ),3,2);
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
    Matrix<float> E1(make_shared_ptr(new float[3*2] {0,0,0.1,0.2,0.3,0.4}),3,2);
    Matrix<float> E2(make_shared_ptr(new float[3*2]{0,0,0.3,0.8,0.2,0.9}),3,2);
    shared_ptr<InputLayer<float>> inputLayer (new InputLayer<float>());
    Matrix<float> W(make_shared_ptr(new float[2*8] {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6}), 2,8);
    Vector<float> b(make_shared_ptr(new float[2]{0.1,0.2}),2);
    std::vector<std::shared_ptr<Layer<float, Vector<float>>>> layers={
        std::shared_ptr<Layer<float, Vector<float>>>(new HiddenLayer<float> (W,b,ActivationFunctions<float>::get(0)))
    };
    SequenceInput <float> sequenceInput({0,2},1,E1,Poolings<Vector<float>>::get(0));
    NonSequenceInput<float> nonsequenceInput(2,E2);
    vector<reference_wrapper<Input<float>>> inputs={sequenceInput,nonsequenceInput};
    MLP<float> nn(inputLayer, layers);
    auto r=nn.calc(inputs);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] ,tanh(0.05*0.1 + 0.1*0.2+ 0.15*0.3 + 0.2*0.4 + 0.2*0.5 + 0.3*0.6 + 0.2*0.7+0.9*0.8 + 0.1)), "r");
    ASSERT(equals(r.data().get()[1] ,tanh(0.05*0.9 + 0.1*1.0+ 0.15*1.1 + 0.2*1.2 + 0.2*1.3 + 0.3*1.4 + 0.2*1.5+0.9*1.6 + 0.2)), "r");
}

void writeReadFilleTest(){
    const char* file="test.bin";
    ofstream os(file,ios::binary);
    float f1=3.26f;
    size_t size1=5;
    os.write((const char*)&f1,sizeof(float));
    os.write((const char*)&size1,sizeof(size_t));
    os.close();
    ifstream is(file,ios::binary);
    float f2=0.0f;
    size_t size2=0;
    is.read((char*)&f2,sizeof(float));
    is.read((char*)&size2,sizeof(size_t));
    ASSERT(equals(f1,f2),"f1,f2");
    ASSERT(equals(size1,size2),"size1,size2");
}

void MLPModelTest(){
    //describe model in memory
    vector<shared_ptr<Matrix<float>>> embeddings={newMatrix(new float[3*2] {0,0,0.1,0.2,0.3,0.4},3,2),newMatrix(new float[3*2]{0,0,0.3,0.8,0.2,0.9},3,2)};
    auto inputsInfo={newInputInfo(*embeddings[0],1,Poolings<Vector<float>>::AVG),newInputInfo(*embeddings[1])};
    auto weights={newMatrix(new float[2*8] {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6}, 2,8)};
    auto biasVectors={newVector(new float[2]{0.1,0.2},2)};
    vector<size_t> activationFunctionIds={ActivationFunctions<float>::TANH};
    MLPModel<float> model(inputsInfo,embeddings,weights,biasVectors,activationFunctionIds);
    //predict
    vector<vector<size_t>> idsInputs ={{0,2},{2}};
    auto r=model.predict(idsInputs);
    auto t1=tanh(0.05*0.1 + 0.1*0.2+ 0.15*0.3 + 0.2*0.4 + 0.2*0.5 + 0.3*0.6 + 0.2*0.7+0.9*0.8 + 0.1);
    auto t2=tanh(0.05*0.9 + 0.1*1.0+ 0.15*1.1 + 0.2*1.2 + 0.2*1.3 + 0.3*1.4 + 0.2*1.5+0.9*1.6 + 0.2);
    auto o1=exp(t1)/(exp(t1)+exp(t2));
    auto o2=exp(t2)/(exp(t1)+exp(t2));
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0],o1), "r");
    ASSERT(equals(r.data().get()[1],o2), "r");
    //save the model
    const char* modelFile="model.bin";
    model.save(modelFile);
    //load the model
    MLPModel<float> modelLoaded;
    modelLoaded.load(modelFile);
    //apply the model
    r=modelLoaded.predict(idsInputs);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0],o1), "r");
    ASSERT(equals(r.data().get()[1],o2), "r");
}

int main( int argc, const char* argv[] )
{
    vectorPlusTest();
    vectorDivideTest();
    vectorMaxTest();
    vectorApplyTest();
    vectorAggregateTest();
    matrixMultiplyTest();
    hiddenLayerTest();
    sequenceInputTest();
    nonSequenceInputTest();
    inputLayerTest();
    softmaxLayerTest();
    MLPTest();
    MLPModelTest();
    writeReadFilleTest();
}