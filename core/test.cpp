#include "nn.hpp"
#include <random>

using namespace std;
using namespace common;
using namespace nn;
using namespace quantization;

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

template<template<class> class MVM>
void matrixMultiplyTest(){
    Matrix<float> m(make_shared_ptr(new float[5*3]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5,3);
    Vector<float> v(make_shared_ptr(new float[3]{0.1,0.2,0.3}),3);
    auto r=multiply<float,MVM>(m,v);
    ASSERT(r.size() == 5, "r");
    ASSERT(equals(r.data().get()[0] , 1.4f), "r[0]");
    ASSERT(equals(r.data().get()[1] , 3.2f), "r[1]");
    ASSERT(equals(r.data().get()[2] , 7*0.1+8*0.2+9*0.3), "r[2]");
    ASSERT(equals(r.data().get()[3] , 10*0.1+11*0.2+12*0.3), "r[3]");
    ASSERT(equals(r.data().get()[4] , 13*0.1+14*0.2+15*0.3), "r[4]");
}

void activationFunctionTest(){
    auto func=ActivationFunctions<float>::get(ActivationFunctions<float>::TANH);
    auto input=0.23f;
    ASSERT(equals(func(input),tanh(input)),"tanh");
    func=ActivationFunctions<float>::get(ActivationFunctions<float>::RELU);
    input=0.25f;
    ASSERT(equals(func(input),input),"relu");
    ASSERT(equals(func(-input),0),"relu");
    func=ActivationFunctions<float>::get(ActivationFunctions<float>::IDENTITY);
    input=0.25f;
    ASSERT(equals(func(input),input),"identity");
    ASSERT(equals(func(-input),-input),"identity");
}

void hiddenLayerTest(){
    Matrix<float> W(make_shared_ptr(new float[2*3]{1,2,3,4,5,6}),2,3);
    Vector<float> b(make_shared_ptr(new float[2]{1.5,1.8}),2);
    HiddenLayer<float,MatrixVectoryMultiplier> layer(W,b,ActivationFunctions<float>::get(0));
    Vector<float> v(make_shared_ptr(new float[3]{0.1,0.2,0.3}),3);
    auto r=layer.calc(v);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , tanh(2.9f)), "r[0]");
    ASSERT(equals(r.data().get()[1] , tanh(5.0f)), "r[1]");
}

template<typename T>
class InputVectorIterator: public Iterator<Vector<T>>{
    public:
        virtual bool next(Vector<T>& buffer) {
            if(m_pos>=m_vectors.size()){
                return false;
            }
            //copy to the buffer
            memcpy(buffer.data().get(), m_vectors[m_pos].data().get(), sizeof(T)*buffer.size());
            ++m_pos;
            return true;
        }
        InputVectorIterator<T>& reset(){
            m_pos=0;
            return *this;
        }
    public:
        InputVectorIterator(const vector<Vector<T>>& vectors):m_vectors(vectors),m_pos(0){}
    private:
        vector<Vector<T>> m_vectors;
        size_t m_pos;
};

void poolingTest(){
    vector<Vector<float>> vectors={Vector<float>(make_shared_ptr(new float[2]{1.0,2.0}),2),Vector<float>(make_shared_ptr(new float[2]{0.8,3.2}),2)};
    Vector<float> buffer(make_shared_ptr(new float[2]),2);
    Vector<float> output(make_shared_ptr(new float[2]),2);
    InputVectorIterator<float> inputVectors(vectors);
    Poolings<Vector<float>>::get(Poolings<float>::AVG).calc(output,inputVectors,buffer);
    ASSERT(equals(output.data().get()[0],0.9f), "output");
    ASSERT(equals(output.data().get()[1],2.6f), "output");
    Poolings<Vector<float>>::get(Poolings<float>::SUM).calc(output,inputVectors.reset(),buffer);
    ASSERT(equals(output.data().get()[0],1.8f), "output");
    ASSERT(equals(output.data().get()[1],5.2f), "output");
    Poolings<Vector<float>>::get(Poolings<float>::MAX).calc(output,inputVectors.reset(),buffer);
    ASSERT(equals(output.data().get()[0],1.0f), "output");
    ASSERT(equals(output.data().get()[1],3.2f), "output");
}

void sequenceInputTest(){
    Matrix<float> E(make_shared_ptr(new float[3*2]{0,0,0.1,0.2,0.3,0.4}),3,2);
    auto pEmbedding=EmbeddingWithRawValues<float>::create(E);
    std::vector<size_t> idSequence = { 0,2 };
    SequenceInput<float> input(idSequence,1,pEmbedding,Poolings<Vector<float>>::get(0));
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
    auto pEmbedding=EmbeddingWithRawValues<float>::create(E);
    NonSequenceInput<float> input(2,pEmbedding);
    auto r=input.get();
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0] , 0.3f), "r");
    ASSERT(equals(r.data().get()[1] , 0.4f), "r");
}

void inputLayerTest(){
    Matrix<float> E1(make_shared_ptr(new float[3*2]{0,0,0.1,0.2,0.3,0.4} ),3,2);
    std::vector<size_t> idSequence = { 0,2 };
    SequenceInput<float> sequenceInput(idSequence,1,EmbeddingWithRawValues<float>::create(E1),Poolings<Vector<float>>::get(0));
    Matrix<float> E2(make_shared_ptr(new float[3*2]{0,0,0.3,0.8,0.2,0.9} ),3,2);
    NonSequenceInput<float> nonSequenceInput(2,EmbeddingWithRawValues<float>::create(E2));
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
        std::shared_ptr<Layer<float, Vector<float>>>(new HiddenLayer<float,MatrixVectoryMultiplier> (W,b,ActivationFunctions<float>::get(0)))
    };
    SequenceInput <float> sequenceInput({0,2},1,EmbeddingWithRawValues<float>::create(E1),Poolings<Vector<float>>::get(0));
    NonSequenceInput<float> nonsequenceInput(2,EmbeddingWithRawValues<float>::create(E2));
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

void cacheTest(){
    auto item=make_shared_ptr(new float[3*2] {0,0,0.1,0.2,0.3,0.4});
    auto key="some_key";
    Cache<float> cache;
    ASSERT(cache.get(key)==nullptr,"cache");
    cache.put(key,item);
    ASSERT(cache.get(key).get()==item.get(),"cached");
}

template<typename T>
class VectorIterator: public Iterator<T>{
    public:
        virtual bool next(T& buffer) {
            if(m_pos>=m_vector.size()){
                return false;
            }
            buffer=m_vector.data().get()[m_pos];
            ++m_pos;
            return true;
        }
        VectorIterator<T>& reset(){
            m_pos=0;
            return *this;
        }
    public:
       VectorIterator(const Vector<T>& vector):m_vector(vector),m_pos(0){}
    private:
        Vector<T> m_vector;
        size_t m_pos;
};

void calcMinMaxTest(){
    auto vector=newVector(new float[5]{1,2,3,-1,8},5);
    VectorIterator<float> it(*vector);
    float min,max;
    calcMinMax(it,min,max);
    ASSERT(equals(min,-1),"min");
    ASSERT(equals(max,8),"max");
}

void quantizationTest(){
    auto vector=newVector(new float[5]{1,2,3,-1,8},5);
    VectorIterator<float> it(*vector);
    float min,max;
    calcMinMax(it,min,max);
    it.reset();
    auto size=20;
    auto quantizer=_16BitsLinearQuantizer<float>::create(it,size,min,max);
    it.reset();
    float cur;
    while(it.next(cur)){
        ASSERT(equals(cur, quantizer->unquantize(quantizer->quantize(cur))),"quantization");
    }
    //more tests
    it.reset();
    auto quantizer2=_8BitsLinearQuantizer<float>::create(it,1,min,max);
    it.reset();
    while(it.next(cur)){
        ASSERT(0==quantizer2->quantize(cur),"quantize");
    }
    ASSERT(equals(13.0/5,quantizer2->unquantize(0)),"unquantize");
}

void embeddingTest(){
    vector<shared_ptr<Matrix<float>>> embeddings={newMatrix(new float[3*2] {0,0,0.1,0.2,0.3,0.4},3,2)};
    auto pEmbedding=EmbeddingWithRawValues<float>::create(*embeddings[0]);
    ASSERT(3==pEmbedding->count(),"count");
    ASSERT(2==pEmbedding->dimension(),"dimension");
    auto buffer=make_shared_ptr(new float[2]);
    pEmbedding->get(0,buffer.get());
    ASSERT(buffer.get()[0]==0,"buffer");
    ASSERT(buffer.get()[1]==0,"buffer");
    pEmbedding->get(1,buffer.get());
    ASSERT(equals(buffer.get()[0],0.1),"buffer");
    ASSERT(equals(buffer.get()[1],0.2),"buffer");
    pEmbedding->get(2,buffer.get());
    ASSERT(equals(buffer.get()[0],0.3),"buffer");
    ASSERT(equals(buffer.get()[1],0.4),"buffer");
}

void quantizedEmbeddingTest(){
    vector<shared_ptr<Matrix<float>>> embeddings={newMatrix(new float[3*2] {0,0,0.1,0.2,0.3,0.3002},3,2)};
    //255
    auto pEmbedding=EmbeddingWithQuantizedValues<float,unsigned char>::create(*embeddings[0]);
    ASSERT(3==pEmbedding->count(),"count");
    ASSERT(2==pEmbedding->dimension(),"dimension");
    auto buffer=make_shared_ptr(new float[2]);
    pEmbedding->get(0,buffer.get());
    ASSERT(equals(buffer.get()[0],0),"buffer");
    ASSERT(equals(buffer.get()[1],0),"buffer");
    pEmbedding->get(1,buffer.get());
    ASSERT(equals(buffer.get()[0],0.1),"buffer");
    ASSERT(equals(buffer.get()[1],0.2),"buffer");
    pEmbedding->get(2,buffer.get());
    ASSERT(equals(buffer.get()[0],0.3001),"buffer");
    ASSERT(equals(buffer.get()[1],0.3001),"buffer");
}

void MLPModelTest(){
    //describe model in memory
    vector<shared_ptr<Matrix<float>>> embeddings={newMatrix(new float[3*2] {0,0,0.1,0.2,0.3,0.4},3,2),newMatrix(new float[3*2]{0,0,0.3,0.8,0.2,0.9},3,2)};
    auto pEmbedding1=EmbeddingWithRawValues<float>::create(*embeddings[0]);
    auto pEmbedding2=EmbeddingWithRawValues<float>::create(*embeddings[1]);
    auto inputsInfo={newInputInfo(pEmbedding1,1,Poolings<Vector<float>>::AVG),newInputInfo(pEmbedding2)};
    //two hidden layers
    auto weights={newMatrix(new float[2*8] {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6}, 2,8),newMatrix(new float[2*2] {0.2,0.3,0.1,0.5}, 2,2)};
    auto biasVectors={newVector(new float[2]{0.1,0.2},2),newVector(new float[2]{0.3,0.4},2)};
    vector<size_t> activationFunctionIds={ActivationFunctions<float>::TANH,ActivationFunctions<float>::TANH};
    auto oh11=tanh(0.05*0.1 + 0.1*0.2+ 0.15*0.3 + 0.2*0.4 + 0.2*0.5 + 0.3*0.6 + 0.2*0.7+0.9*0.8 + 0.1);
    auto oh12=tanh(0.05*0.9 + 0.1*1.0+ 0.15*1.1 + 0.2*1.2 + 0.2*1.3 + 0.3*1.4 + 0.2*1.5+0.9*1.6 + 0.2);
    auto t1=tanh(oh11*0.2 + oh12*0.3+0.3);
    auto t2=tanh(oh11*0.1 + oh12*0.5+0.4);
    //soft max
    auto o1=exp(t1)/(exp(t1)+exp(t2));
    auto o2=exp(t2)/(exp(t1)+exp(t2));
    MLPModel<float> model(inputsInfo,weights,biasVectors,activationFunctionIds);
    //predict
    vector<vector<size_t>> idsInputs ={{0,2},{2}};
    auto r=model.predict(idsInputs);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0],o1), "r");
    ASSERT(equals(r.data().get()[1],o2), "r");
    //save and load model
    const char* modelFile="model.bin";
    MLPModelFactory<float>::save(modelFile,inputsInfo,weights,biasVectors,activationFunctionIds);
    auto pModelLoaded=MLPModelFactory<float>::load(modelFile);
    r=pModelLoaded->predict(idsInputs);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0],o1), "r");
    ASSERT(equals(r.data().get()[1],o2), "r");
    //load the model with embedding cached
    auto pModelWithCache=MLPModelFactory<float>::load(modelFile);
    //apply the model
    r=pModelWithCache->predict(idsInputs);
    ASSERT(r.size() == 2, "r");
    ASSERT(equals(r.data().get()[0],o1), "r");
    ASSERT(equals(r.data().get()[1],o2), "r");
}

template<typename T>
void generateRandomNumbers(T* buffer, size_t size, T min=0, T max=1000){
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<T> dist(min,max);
    for(size_t i=0;i<size;++i){
        buffer[i]=dist(mt);
    }
}

void generateRandomNumbers(vector<size_t>& buffer, size_t size, size_t min, size_t max){
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<size_t> dist(min,max);
    for(size_t i=0;i<size;++i){
        buffer[i]=dist(mt);
    }
}

//Defines helper function to create shared pointer of vector without using external data buffer.
template<typename T>
shared_ptr<Vector<T>> newVector(size_t size){
    return make_shared_ptr(new Vector<T>( make_shared_ptr(new T[size]),size));
}

//Defines helper function to create shared pointer of matrix, without using external data buffer.
template<typename T>
shared_ptr<Matrix<T>> newMatrix(size_t row, size_t col){
    return make_shared_ptr(new Matrix<T>(make_shared_ptr(new T[row*col]),row,col));
}

void perfTestWithBigFakedModelSetup(const string& modelFile,size_t numberOfWords, size_t numberOfOther){
    auto dimensionOfWordEmbedding=50;
    auto dimensionOfOtherEmbedding=20;
    auto wordEmbedding=newMatrix<float>(numberOfWords,dimensionOfWordEmbedding);
    generateRandomNumbers(wordEmbedding->data().get(),numberOfWords*dimensionOfWordEmbedding);
    auto otherEmbedding=newMatrix<float>(numberOfOther,dimensionOfOtherEmbedding);
    generateRandomNumbers(otherEmbedding->data().get(),numberOfOther*dimensionOfOtherEmbedding);
    auto contextLength=1;
    auto inputsInfo={newInputInfo(EmbeddingWithRawValues<float>::create(*wordEmbedding),contextLength,Poolings<Vector<float>>::AVG),newInputInfo(EmbeddingWithRawValues<float>::create(*otherEmbedding))};
    //two hidden layers
    auto hiddenLayer0NumberOfOutputNodes=60;
    auto hiddenLayer1NumberOfOutputNodes=18;
    vector<shared_ptr<Matrix<float>>> weights={newMatrix<float>(hiddenLayer0NumberOfOutputNodes,(2*contextLength+1)*dimensionOfWordEmbedding+dimensionOfOtherEmbedding),newMatrix<float>(hiddenLayer1NumberOfOutputNodes,hiddenLayer0NumberOfOutputNodes)};
    vector<shared_ptr<Vector<float>>> biasVectors={newVector<float>(hiddenLayer0NumberOfOutputNodes),newVector<float>(hiddenLayer1NumberOfOutputNodes)};
    generateRandomNumbers(weights[0]->data().get(),hiddenLayer0NumberOfOutputNodes*((2*contextLength+1)*dimensionOfWordEmbedding+dimensionOfOtherEmbedding));
    generateRandomNumbers(weights[1]->data().get(),hiddenLayer1NumberOfOutputNodes*hiddenLayer0NumberOfOutputNodes);
    generateRandomNumbers(biasVectors[0]->data().get(),hiddenLayer0NumberOfOutputNodes);
    generateRandomNumbers(biasVectors[1]->data().get(),hiddenLayer1NumberOfOutputNodes);
    size_t activationFunctionId=ActivationFunctions<float>::RELU;
    vector<size_t> activationFunctionIds={activationFunctionId,activationFunctionId};
    MLPModelFactory<float>::save(modelFile,inputsInfo,weights,biasVectors,activationFunctionIds);
}

void perfTestWithBigFakedModel(const string& modelFile,size_t numberOfWords, size_t numberOfOther){ 
    perfTestWithBigFakedModelSetup(modelFile,numberOfWords,numberOfOther);
    auto pModel=MLPModelFactory<float>::load(modelFile);
    auto predictionTimes=1000;
    auto sequenceLength=25;
    vector<size_t> wordIdSequence(sequenceLength);
    vector<size_t> otherId(1);
    #ifdef PERF
    auto wctstart=CLOCK::now();
    #endif
    for(auto i=0;i<predictionTimes;++i){
        generateRandomNumbers(wordIdSequence,sequenceLength,0,numberOfWords-1);
        generateRandomNumbers(otherId,1,0,numberOfOther-1);
        pModel->predict({wordIdSequence,otherId});
    }
    #ifdef PERF
    auto wctduration = (CLOCK::now()-wctstart);
    cout << "PERF\tfinished in " << microseconds(wctduration) << " micro seconds (Wall Clock)" << endl;
    #endif
}

template<template<class> class MVM>
void perfTestWithRealModel(){
    //[ 411 8286 4659 ][ 4 14 8 ][ 64 34 869 ][ 56 29 59 ][ 131 204 59 ][ 226 529 508 ][ 9 6 14 ][ 2 ]
    auto pModel=MLPModelFactory<float>::load<EmbeddingWithRawValues,MVM>("zoe_random_nbow-81.model.bin");
    vector<vector<size_t>> idsInputs={{411,8286,4659},{4,14,8},{64,34,869},{56,29,59},{131,204,59},{226,529,508},{9,6,14},{2}};
    auto predictionTimes=1000;
    #ifdef PERF
    auto wctstart=CLOCK::now();
    #endif
    for(auto i=0;i<predictionTimes;++i){
        pModel->predict(idsInputs);
    }
    #ifdef PERF
    auto wctduration = (CLOCK::now()-wctstart);
    cout << "PERF\tfinished in " << microseconds(wctduration) << " micro seconds (Wall Clock)" << endl;
    #endif
}

//Defines A*x implementation
template <class T>
class MatrixVectoryMultiplierBaseline{
    public:
        static inline void multiply(const T* A, const T* x, T* y, const size_t row, const size_t col){
            ASSERT(A,"A");
            ASSERT(x,"x");
            ASSERT(y,"y");
            ASSERT(row>0,"row");
            ASSERT(col>0,"col");
            const T* a=A;
            const T* _x;
            T y1;
            for(size_t i=0;i<row;++i){
                _x=x;
                y1=0;
                for(size_t j=0;j<col;++j){
                    y1+=(*_x++)*(*a++);
                }
                *y++=y1;
            }
        }
};

#define DOT4(x,y)  (x[0]*y[0]+x[1]*y[1]+x[2]*y[2]+x[3]*y[3])
#define DOT(x,y)  ((*x)*(*y))

template <class T>
class MatrixVectoryMultiplierMoreUnRolling{
    public:
        static inline void multiply(const T* A, const T* x, T* y, const size_t row, const size_t col){
            ASSERT(A,"A");
            ASSERT(x,"x");
            ASSERT(y,"y");
            ASSERT(row>0,"row");
            ASSERT(col>0,"col");
            const T* a1=A, *a2=A+col,*a3=A+2*col,*a4=A+3*col,*_x;
            T y1,y2,y3,y4;
            size_t i=0,j;
            //process 4 rows per loop
            for(;i+3<row;i+=4,y+=4,a1+=3*col,a2+=3*col,a3+=3*col,a4+=3*col){
                for(_x=x,y1=0,y2=0,y3=0,y4=0,j=0;j+3<col;j+=4,_x+=4,a1+=4,a2+=4,a3+=4,a4+=4){
                    y1+=DOT4(_x,a1);
                    y2+=DOT4(_x,a2);
                    y3+=DOT4(_x,a3);
                    y4+=DOT4(_x,a4);
                }
                for(;j<col;++j,++_x,++a1,++a2,++a3,++a4){
                    y1+=DOT(_x,a1);
                    y2+=DOT(_x,a2);
                    y3+=DOT(_x,a3);
                    y4+=DOT(_x,a4);
                }
                y[0]=y1,y[1]=y2,y[2]=y3,y[3]=y4;
            }
            if(row-i==3){
                for(_x=x,y1=0,y2=0,y3=0,j=0;j+3<col;j+=4,_x+=4,a1+=4,a2+=4,a3+=4){
                    y1+=DOT4(_x,a1);
                    y2+=DOT4(_x,a2);
                    y3+=DOT4(_x,a3);
                }
                for(;j<col;++j,++_x,++a1,++a2,++a3){
                    y1+=DOT(_x,a1);
                    y2+=DOT(_x,a2);
                    y3+=DOT(_x,a3);
                }
                y[0]=y1,y[1]=y2,y[2]=y3;
            }else if(row-i==2){
                for(_x=x,y1=0,y2=0,j=0;j+3<col;j+=4,_x+=4,a1+=4,a2+=4){
                    y1+=DOT4(_x,a1);
                    y2+=DOT4(_x,a2);
                }
                for(;j<col;++j,++_x,++a1,++a2){
                    y1+=DOT(_x,a1);
                    y2+=DOT(_x,a2);
                }
                y[0]=y1,y[1]=y2;
            }else if(row-i==1){
                for(_x=x,y1=0,j=0;j+3<col;j+=4,_x+=4,a1+=4){
                    y1+=DOT4(_x,a1);
                }
                for(;j<col;++j,++_x,++a1){
                    y1+=DOT(_x,a1);
                }
                y[0]=y1;
            }
        }
};

void unitTest(){
    vectorPlusTest();
    vectorDivideTest();
    vectorMaxTest();
    vectorApplyTest();
    vectorAggregateTest();    
    matrixMultiplyTest<MatrixVectoryMultiplierBaseline>();
    matrixMultiplyTest<MatrixVectoryMultiplier>();
    matrixMultiplyTest<MatrixVectoryMultiplierMoreUnRolling>();
    activationFunctionTest();
    poolingTest();
    hiddenLayerTest();
    sequenceInputTest();
    nonSequenceInputTest();
    inputLayerTest();
    softmaxLayerTest();
    writeReadFilleTest();
    calcMinMaxTest();
    quantizationTest();
    cacheTest();
    embeddingTest();
    quantizedEmbeddingTest();
    MLPTest();
    MLPModelTest();
}

void perfTest(){
    //1 million words
    auto numberOfWords=1000000;
    auto numberOfOther=1000;
    string modelFile="model.faked.bin";
    //perfTestWithBigFakedModelSetup(modelFile,numberOfWords,numberOfOther);
    perfTestWithBigFakedModel(modelFile,numberOfWords,numberOfOther);
    perfTestWithBigFakedModel(modelFile,numberOfWords,numberOfOther);
}

int main( int argc, const char* argv[] )
{
    string option;
    if(argc==2){
        option=argv[1];
    }
    if(option=="perf"){
        perfTest();
    }
    if(option=="perfReal"){
        perfTestWithRealModel<MatrixVectoryMultiplierBaseline>();
        perfTestWithRealModel<MatrixVectoryMultiplier>();
        perfTestWithRealModel<MatrixVectoryMultiplierMoreUnRolling>();
    }
    else if(option=="all"){
        unitTest();
        perfTest();
        perfTestWithRealModel<MatrixVectoryMultiplierBaseline>();
        perfTestWithRealModel<MatrixVectoryMultiplier>();
        perfTestWithRealModel<MatrixVectoryMultiplierMoreUnRolling>();
    } else{
        //default: do unit test
        unitTest();
    }
}
