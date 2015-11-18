#include "nn.hpp"
#include "nn.test.hpp"

using namespace std;
using namespace common;
using namespace nn::test;
using namespace nn;
using namespace quantization;
using namespace boost;


template<typename T>
void vectorPlusTest(){
    DECL_VECTOR(T,v1,2,{0.5,0.5})
    DECL_VECTOR(T,v,2,{0.2,0.3})
    v.plus(v1);
    ASSERT_EQUALS(T,v,2,{0.7,0.8});
}

template<typename T>
void vectorDivideTest(){
    DECL_VECTOR(T,v,2,{0.2,0.4})
    v.divide(2);
    ASSERT_EQUALS(T,v,2,{0.1,0.2});
}

template<typename T>
void vectorMaxTest(){
    DECL_VECTOR(T,v1,2,{0.3,0.4})
    DECL_VECTOR(T,v,2,{0.5,0.1})
    v.max(v1);
    ASSERT_EQUALS(T,v,2,{0.5,0.4});
}

template<typename T>
T inc(const T& value){
    return value+1;
}

template<typename T>
void vectorApplyTest(){
    DECL_VECTOR(T,v,2,{0.5,0.6})
    v.apply(inc<T>);
    ASSERT_EQUALS(T,v,2,{1.5,1.6});
}

template<typename T>
T sum(const T& t1, const T& t2){
    return t1+t2;
}

template<typename T>
void vectorAggregateTest(){
    DECL_VECTOR(T,v,2,{0.5,0.6})
    ASSERT(equals(v.aggregate(sum<T>,0) , 1.1), "result");
}

template<typename T,template<class> class MVM>
void matrixMultiplyTest(){
    DECL_MATRIX(T,m,5,3,{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15})
    DECL_VECTOR(T,v,3,{0.1,0.2,0.3})
    Vector<T> r=multiply<T,MVM>(m,v);
    ASSERT_EQUALS(T,r,5,{1.4,3.2,7*0.1+8*0.2+9*0.3,10*0.1+11*0.2+12*0.3,13*0.1+14*0.2+15*0.3});
}

//To cover the special case that n%4=3
template<typename T,template<class> class MVM>
void matrixMultiplyEdgeTest(){
    DECL_MATRIX(T,m,3,3,{1,2,3,4,5,6,7,8,9})
    DECL_VECTOR(T,v,3,{0.1,0.2,0.3})
    Vector<T> r=multiply<T,MVM>(m,v);
    ASSERT_EQUALS(T,r,3,{1.4,3.2,7*0.1+8*0.2+9*0.3});
}

template<typename T>
void activationFunctionTest(){
    T input=0.23;
    ASSERT(equals(ActivationFunctions<T>::get(ActivationFunctions<T>::TANH)(input),tanh(input)),"tanh");
    input=0.25;
    ASSERT(equals(ActivationFunctions<T>::get(ActivationFunctions<T>::RELU)(input),input),"relu");
    ASSERT(equals(ActivationFunctions<T>::get(ActivationFunctions<T>::RELU)(-input),0),"relu");
    input=0.25;
    ASSERT(equals(ActivationFunctions<T>::get(ActivationFunctions<T>::IDENTITY)(input),input),"identity");
    ASSERT(equals(ActivationFunctions<T>::get(ActivationFunctions<T>::IDENTITY)(-input),-input),"identity");
}

template<typename T>
void hiddenLayerTest(){
    DECL_MATRIX(T,W,2,3,{1,2,3,4,5,6})
    DECL_VECTOR(T,b,2,{1.5,1.8})
    HiddenLayer<T,MatrixVectoryMultiplier> layer(W,b,ActivationFunctions<T>::get(0));
    DECL_VECTOR(T,v,3,{0.1,0.2,0.3})
    Vector<T> r=layer.calc(v);
    ASSERT_EQUALS(T,r,2,{tanh(2.9),tanh(5.0)});
}

template<typename T>
class InputVectorIterator: public Iterator<Vector<T> >{
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
        InputVectorIterator(const vector<Vector<T> >& vectors):m_vectors(vectors),m_pos(0){}
    private:
        vector<Vector<T> > m_vectors;
        size_t m_pos;
};

template<typename T>
void poolingTest(){
    DECL_VECTOR(T,v1,2,{1.0,2.0})
    DECL_VECTOR(T,v2,2,{0.8,3.2})
    vector<Vector<T> > vectors;
    vectors.push_back(v1);
    vectors.push_back(v2);
    Vector<T> buffer(make_shared_ptr(new T[2]),2);
    Vector<T> output(make_shared_ptr(new T[2]),2);
    InputVectorIterator<T> inputVectors(vectors);
    Poolings<Vector<T> >::get(Poolings<T>::AVG).calc(output,inputVectors,buffer);
    ASSERT_EQUALS(T,output,2,{0.9,2.6});
    Poolings<Vector<T> >::get(Poolings<T>::SUM).calc(output,inputVectors.reset(),buffer);
    ASSERT_EQUALS(T,output,2,{1.8,5.2});
    Poolings<Vector<T> >::get(Poolings<T>::MAX).calc(output,inputVectors.reset(),buffer);
    ASSERT_EQUALS(T,output,2,{1.0,3.2});
}

template<typename T>
void sequenceInputTest(){
    DECL_MATRIX(T,E,3,2,{0,0,0.1,0.2,0.3,0.4})
    shared_ptr<Embedding<T> > pEmbedding=EmbeddingWithRawValues<T>::create(E);
    vector<size_t> idSequence;
    idSequence.push_back(0);
    idSequence.push_back(2);
    SequenceInput<T> input(idSequence,1,pEmbedding,Poolings<Vector<T> >::get(0));
    Vector<T> r=input.get();
    ASSERT_EQUALS(T,r,2*3,{0.05,0.1,0.15,0.2,0.2,0.3});
}

template<typename T>
void nonSequenceInputTest(){
    DECL_MATRIX(T,E,3,2,{0,0,0.1,0.2,0.3,0.4})
    shared_ptr<Embedding<T> > pEmbedding=EmbeddingWithRawValues<T>::create(E);
    NonSequenceInput<T> input(2,pEmbedding);
    Vector<T> r=input.get();
    ASSERT_EQUALS(T,r,2,{0.3,0.4});
}

template<typename T>
void inputLayerTest(){
    DECL_MATRIX(T,E1,3,2,{0,0,0.1,0.2,0.3,0.4})
    vector<size_t> idSequence;
    idSequence.push_back(0);
    idSequence.push_back(2);
    shared_ptr<Input<T> > sequenceInput=make_shared_ptr(new SequenceInput<T>(idSequence,1,EmbeddingWithRawValues<T>::create(E1),Poolings<Vector<T> >::get(0)));
    DECL_MATRIX(T,E2,3,2,{0,0,0.3,0.8,0.2,0.9})
    shared_ptr<Input<T> > nonSequenceInput=make_shared_ptr(new NonSequenceInput<T>(2,EmbeddingWithRawValues<T>::create(E2)));
    vector<shared_ptr<Input<T> > > inputs;
    inputs.push_back(sequenceInput);
    inputs.push_back(nonSequenceInput);
    InputLayer<T> layer;
    Vector<T> r =layer.calc(inputs);
    ASSERT_EQUALS(T,r,2*3+2,{0.05,0.1,0.15,0.2,0.2,0.3,0.2,0.9});
}

template<typename T>
void softmaxLayerTest(){
    DECL_VECTOR(T,input,2,{0.5,0.5})
    SoftmaxLayer<T> softmaxLayer;
    Vector<T> result = softmaxLayer.calc(input);
    ASSERT_EQUALS(T,result,2,{0.5,0.5});
}



template<typename T>
void MLPTest(){
    DECL_MATRIX(T,E1,3,2,{0,0,0.1,0.2,0.3,0.4})
    DECL_MATRIX(T,E2,3,2,{0,0,0.3,0.8,0.2,0.9})
    shared_ptr<InputLayer<T> > inputLayer (new InputLayer<T>());
    DECL_MATRIX(T,W,2,8,{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6})
    DECL_VECTOR(T,b,2,{0.1,0.2})
    vector<shared_ptr<Layer<T, Vector<T> > > > layers;
    layers.push_back(shared_ptr<Layer<T, Vector<T> > >(new HiddenLayer<T,MatrixVectoryMultiplier> (W,b,ActivationFunctions<T>::get(0))));
    DECL_STD_VECTOR(size_t,idSequence,2,{0,2})
    shared_ptr<Input<T> > sequenceInput=make_shared_ptr(new SequenceInput<T>(idSequence,1,EmbeddingWithRawValues<T>::create(E1),Poolings<Vector<T> >::get(0)));
    shared_ptr<Input<T> > nonSequenceInput=make_shared_ptr(new NonSequenceInput<T>(2,EmbeddingWithRawValues<T>::create(E2)));
    vector<shared_ptr<Input<T> > > inputs;
    inputs.push_back(sequenceInput);
    inputs.push_back(nonSequenceInput);
    MLP<T> nn(inputLayer, layers);
    Vector<T> r=nn.calc(inputs);
    ASSERT_EQUALS(T,r,2,{tanh(0.05*0.1 + 0.1*0.2+ 0.15*0.3 + 0.2*0.4 + 0.2*0.5 + 0.3*0.6 + 0.2*0.7+0.9*0.8 + 0.1),tanh(0.05*0.9 + 0.1*1.0+ 0.15*1.1 + 0.2*1.2 + 0.2*1.3 + 0.3*1.4 + 0.2*1.5+0.9*1.6 + 0.2)});
}

template<typename T>
void writeReadFilleTest(){
    const char* file="test.bin";
    ofstream os(file,ios::binary);
    T f1=3.26;
    size_t size1=5;
    os.write((const char*)&f1,sizeof(T));
    os.write((const char*)&size1,sizeof(size_t));
    os.close();
    ifstream is(file,ios::binary);
    T f2=0.0;
    size_t size2=0;
    is.read((char*)&f2,sizeof(T));
    is.read((char*)&size2,sizeof(size_t));
    ASSERT(equals(f1,f2),"f1,f2");
    ASSERT(equals(size1,size2),"size1,size2");
}

template<typename T>
void cacheTest(){
    shared_ptr<T> item=make_shared_ptr(new T[3*2]);
    string key="some_key";
    Cache<T> cache;
    ASSERT(!cache.get(key),"cache");
    cache.put(key,item);
    ASSERT(cache.get(key).get()==item.get(),"cached");
}

template<typename T>
void cacheTestForSizet(){
    shared_ptr<T> item=make_shared_ptr(new T[3*2]);
    Cache<T,size_t> cache;
    size_t key=cache.put(item);
    ASSERT(key==1,"key");
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

template<typename T>
void calcMinMaxTest(){
    DECL_VECTOR(T,vector,5,{1,2,3,-1,8})
    VectorIterator<T> it(vector);
    T min,max;
    calcMinMax(it,min,max);
    ASSERT(equals(min,-1),"min");
    ASSERT(equals(max,8),"max");
}

template<typename T>
void quantizationTest(){
    DECL_VECTOR(T,vector,5,{1,2,3,-1,8})
    VectorIterator<T> it(vector);
    T min,max;
    calcMinMax(it,min,max);
    it.reset();
    size_t size=20;
    shared_ptr<Quantizer<T,unsigned short> > quantizer=_16BitsLinearQuantizer<T>::create(it,size,min,max);
    it.reset();
    T cur;
    while(it.next(cur)){
        ASSERT(equals(cur, quantizer->unquantize(quantizer->quantize(cur))),"quantization");
    }
    //more tests
    it.reset();
    shared_ptr<Quantizer<T,unsigned char> > quantizer2=_8BitsLinearQuantizer<T>::create(it,1,min,max);
    it.reset();
    while(it.next(cur)){
        ASSERT(0==quantizer2->quantize(cur),"quantize");
    }
    ASSERT(equals(13.0/5,quantizer2->unquantize(0)),"unquantize");
}

template<typename T>
void embeddingTest(){
    DECL_MATRIX_PTR(T,E,3,2,{0,0,0.1,0.2,0.3,0.4})
    vector<shared_ptr<Matrix<T> > > embeddings;
    embeddings.push_back(E);
    shared_ptr<Embedding<T> > pEmbedding=EmbeddingWithRawValues<T>::create(*embeddings[0]);
    ASSERT(3==pEmbedding->count(),"count");
    ASSERT(2==pEmbedding->dimension(),"dimension");
    shared_ptr<Vector<T> > buffer=newVector(new T[2],2);
    pEmbedding->get(0,buffer->data().get());
    ASSERT_EQUALS(T,*buffer,2,{0,0});
    pEmbedding->get(1,buffer->data().get());
    ASSERT_EQUALS(T,*buffer,2,{0.1,0.2});
    pEmbedding->get(2,buffer->data().get());
    ASSERT_EQUALS(T,*buffer,2,{0.3,0.4});
}

template<typename T>
void quantizedEmbeddingTest(){
    DECL_MATRIX_PTR(T,E,3,2,{0,0,0.1,0.2,0.3,0.3002})
    vector<shared_ptr<Matrix<T> > > embeddings;
    embeddings.push_back(E);
    //255
    shared_ptr<Embedding<T> > pEmbedding=EmbeddingWithQuantizedValues<T,unsigned char>::create(*embeddings[0]);
    ASSERT(3==pEmbedding->count(),"count");
    ASSERT(2==pEmbedding->dimension(),"dimension");
    shared_ptr<Vector<T> > buffer=newVector(new T[2],2);
    pEmbedding->get(0,buffer->data().get());
    ASSERT_EQUALS(T,*buffer,2,{0,0});
    pEmbedding->get(1,buffer->data().get());
    ASSERT_EQUALS(T,*buffer,2,{0.1,0.2});
    pEmbedding->get(2,buffer->data().get());
    ASSERT_EQUALS(T,*buffer,2,{0.3001,0.3001});
}

template<typename T,template<class> class E,template<class> class MVM>
void MLPModelTest(){
    //describe model in memory
    DECL_MATRIX_PTR(T,E1,3,2,{0,0,0.1,0.2,0.3,0.4})
    DECL_MATRIX_PTR(T,E2,3,2,{0,0,0.3,0.8,0.2,0.9})
    vector<shared_ptr<Matrix<T> > > embeddings;
    embeddings.push_back(E1);
    embeddings.push_back(E2);
    shared_ptr<Embedding<T> > pEmbedding1=EmbeddingWithRawValues<T>::create(*embeddings[0]);
    shared_ptr<Embedding<T> > pEmbedding2=EmbeddingWithRawValues<T>::create(*embeddings[1]);
    vector<shared_ptr<InputInfo<T> > > inputsInfo;
    inputsInfo.push_back(newInputInfo(pEmbedding1,1,Poolings<Vector<T> >::AVG));
    inputsInfo.push_back(newInputInfo(pEmbedding2));
    //two hidden layers
    DECL_MATRIX_PTR(T,W1,2,8,{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6})
    DECL_MATRIX_PTR(T,W2,2,2,{0.2,0.3,0.1,0.5})
    vector<shared_ptr<Matrix<T> > > weights;
    weights.push_back(W1);
    weights.push_back(W2);
    DECL_VECTOR_PTR(T,b1,2,{0.1,0.2})
    DECL_VECTOR_PTR(T,b2,2,{0.3,0.4})
    vector<shared_ptr<Vector<T> > > biasVectors;
    biasVectors.push_back(b1);
    biasVectors.push_back(b2);
    vector<size_t> activationFunctionIds;
    activationFunctionIds.push_back(ActivationFunctions<T>::TANH);
    activationFunctionIds.push_back(ActivationFunctions<T>::TANH);
    T oh11=tanh(0.05*0.1 + 0.1*0.2+ 0.15*0.3 + 0.2*0.4 + 0.2*0.5 + 0.3*0.6 + 0.2*0.7+0.9*0.8 + 0.1);
    T oh12=tanh(0.05*0.9 + 0.1*1.0+ 0.15*1.1 + 0.2*1.2 + 0.2*1.3 + 0.3*1.4 + 0.2*1.5+0.9*1.6 + 0.2);
    T t1=tanh(oh11*0.2 + oh12*0.3+0.3);
    T t2=tanh(oh11*0.1 + oh12*0.5+0.4);
    //soft max
    T o1=exp(t1)/(exp(t1)+exp(t2));
    T o2=exp(t2)/(exp(t1)+exp(t2));
    MLPModel<T> model(inputsInfo,weights,biasVectors,activationFunctionIds);
    //predict
    vector<vector<size_t> > idsInputs;
    DECL_STD_VECTOR(size_t,ids1,2,{0,2})
    DECL_STD_VECTOR(size_t,ids2,1,{2})
    idsInputs.push_back(ids1);
    idsInputs.push_back(ids2);
    Vector<T> r=model.predict(idsInputs);
    ASSERT_EQUALS(T,r,2,{o1,o2});
    //save and load model
    string modelFile="model.bin";
    MLPModelFactory::save<T>(modelFile,inputsInfo,weights,biasVectors,activationFunctionIds);
    shared_ptr<MLPModel<T,MVM> > pModelLoaded=MLPModelFactory::load<T,E,MVM>(modelFile);
    r=pModelLoaded->predict(idsInputs);
    ASSERT_EQUALS(T,r,2,{o1,o2});
    //load the model with embedding cached
    shared_ptr<MLPModel<T,MVM> > pModelWithCache=MLPModelFactory::load<T,E,MVM>(modelFile);
    //apply the model
    r=pModelWithCache->predict(idsInputs);
    ASSERT_EQUALS(T,r,2,{o1,o2});
}

template<typename T>
void perfTestWithBigFakedModelSetup(const string& modelFile,size_t numberOfWords, size_t numberOfOther,size_t dimensionOfWordEmbedding, size_t contextLength, size_t dimensionOfOtherEmbedding,size_t hiddenLayer0NumberOfOutputNodes, size_t hiddenLayer1NumberOfOutputNodes){
    //two inputs
    shared_ptr<Matrix<T> > wordEmbedding=newMatrix<T>(numberOfWords,dimensionOfWordEmbedding);
    generateRandomNumbers(wordEmbedding->data().get(),numberOfWords*dimensionOfWordEmbedding);
    shared_ptr<Matrix<T> > otherEmbedding=newMatrix<T>(numberOfOther,dimensionOfOtherEmbedding);
    generateRandomNumbers(otherEmbedding->data().get(),numberOfOther*dimensionOfOtherEmbedding);
    shared_ptr<Embedding<T> > pEmbedding1=EmbeddingWithRawValues<T>::create(*wordEmbedding);
    shared_ptr<Embedding<T> > pEmbedding2=EmbeddingWithRawValues<T>::create(*otherEmbedding);
    vector<shared_ptr<InputInfo<T> > > inputsInfo;
    shared_ptr<InputInfo<T> > inputInfo1=newInputInfo(pEmbedding1,contextLength,Poolings<Vector<T> >::AVG);
    shared_ptr<InputInfo<T> > inputInfo2=newInputInfo(pEmbedding2);
    inputsInfo.push_back(inputInfo1);
    inputsInfo.push_back(inputInfo2);
    //two hidden layers
    shared_ptr<Matrix<T> > W1=newMatrix<T>(hiddenLayer0NumberOfOutputNodes,(2*contextLength+1)*dimensionOfWordEmbedding+dimensionOfOtherEmbedding);
    shared_ptr<Matrix<T> > W2=newMatrix<T>(hiddenLayer1NumberOfOutputNodes,hiddenLayer0NumberOfOutputNodes);
    vector<shared_ptr<Matrix<T> > > weights;
    weights.push_back(W1);
    weights.push_back(W2);
    generateRandomNumbers(weights[0]->data().get(),hiddenLayer0NumberOfOutputNodes*((2*contextLength+1)*dimensionOfWordEmbedding+dimensionOfOtherEmbedding));
    generateRandomNumbers(weights[1]->data().get(),hiddenLayer1NumberOfOutputNodes*hiddenLayer0NumberOfOutputNodes);
    shared_ptr<Vector<T> > b1=newVector<T>(hiddenLayer0NumberOfOutputNodes);
    shared_ptr<Vector<T> > b2=newVector<T>(hiddenLayer1NumberOfOutputNodes);
    vector<shared_ptr<Vector<T> > > biasVectors;
    biasVectors.push_back(b1);
    biasVectors.push_back(b2);
    generateRandomNumbers(biasVectors[0]->data().get(),hiddenLayer0NumberOfOutputNodes);
    generateRandomNumbers(biasVectors[1]->data().get(),hiddenLayer1NumberOfOutputNodes);
    vector<size_t> activationFunctionIds;
    activationFunctionIds.push_back(ActivationFunctions<T>::RELU);
    activationFunctionIds.push_back(ActivationFunctions<T>::RELU);
    MLPModelFactory::save<T>(modelFile,inputsInfo,weights,biasVectors,activationFunctionIds);
}

template<typename T,template<class> class E,template<class> class MVM>
void perfTestWithBigFakedModel(const string& modelFile,size_t numberOfWords, size_t numberOfOther){ 
    shared_ptr<MLPModel<T,MVM> > pModel=MLPModelFactory::load<T,E,MVM>(modelFile);
    size_t predictionTimes=1000;
    size_t sequenceLength=25;
    vector<size_t> wordIdSequence(sequenceLength);
    vector<size_t> otherId(1);
    #ifdef PERF
    clock_t wctstart=clock();
    #endif
    for(size_t i=0;i<predictionTimes;++i){
        generateRandomNumbers(wordIdSequence,sequenceLength,0,numberOfWords-1);
        generateRandomNumbers(otherId,1,0,numberOfOther-1);
        vector<vector<size_t> > idsInputs;
        idsInputs.push_back(wordIdSequence);
        idsInputs.push_back(otherId);
        pModel->predict(idsInputs);
    }
    #ifdef PERF
    clock_t wctduration = (clock()-wctstart);
    cout << "PERF\tfinished in " << microseconds(wctduration) << " micro seconds (Wall Clock)" << endl;
    #endif
}

template<template<class> class E,template<class> class MVM>
void perfTestWithRealModel(){
    //[ 411 8286 4659 ][ 4 14 8 ][ 64 34 869 ][ 56 29 59 ][ 131 204 59 ][ 226 529 508 ][ 9 6 14 ][ 2 ]
    shared_ptr<MLPModel<float,MVM> > pModel=MLPModelFactory::load<float,E,MVM>("zoe_random_nbow-81.model.bin");
    vector<vector<size_t> > idsInputs;
    DECL_STD_VECTOR(size_t,ids1,3,{411,8286,4659})
    DECL_STD_VECTOR(size_t,ids2,3,{4,14,8})
    DECL_STD_VECTOR(size_t,ids3,3,{64,34,869})
    DECL_STD_VECTOR(size_t,ids4,3,{56,29,59})
    DECL_STD_VECTOR(size_t,ids5,3,{131,204,59})
    DECL_STD_VECTOR(size_t,ids6,3,{226,529,508})
    DECL_STD_VECTOR(size_t,ids7,3,{9,6,14})
    DECL_STD_VECTOR(size_t,ids8,1,{2})
    INIT_STD_VECTOR(vector<size_t>,idsInputs,8,{ids1,ids2,ids3,ids4,ids5,ids6,ids7,ids8})
    size_t predictionTimes=1000;
    #ifdef PERF
    clock_t wctstart=clock();
    #endif
    for(size_t i=0;i<predictionTimes;++i){
        pModel->predict(idsInputs);
    }
    #ifdef PERF
    clock_t wctduration = (clock()-wctstart);
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

template<class T>
void unitTest(){
    vectorPlusTest<T>();
    vectorDivideTest<T>();
    vectorMaxTest<T>();
    vectorApplyTest<T>();
    vectorAggregateTest<T>();
    //different matrix vector multiplier
    matrixMultiplyTest<T,MatrixVectoryMultiplierBaseline>();
    matrixMultiplyTest<T,MatrixVectoryMultiplier>();
    matrixMultiplyTest<T,MatrixVectoryMultiplierMoreUnRolling>();
    matrixMultiplyEdgeTest<T,MatrixVectoryMultiplierBaseline>();
    matrixMultiplyEdgeTest<T,MatrixVectoryMultiplier>();
    matrixMultiplyEdgeTest<T,MatrixVectoryMultiplierMoreUnRolling>();
    activationFunctionTest<T>();
    poolingTest<T>();
    hiddenLayerTest<T>();
    sequenceInputTest<T>();
    nonSequenceInputTest<T>();
    inputLayerTest<T>();
    softmaxLayerTest<T>();
    writeReadFilleTest<T>();
    calcMinMaxTest<T>();
    quantizationTest<T>();
    cacheTest<T>();
    embeddingTest<T>();
    quantizedEmbeddingTest<T>();
    MLPTest<T>();
    MLPModelTest<T,EmbeddingWithRawValues,MatrixVectoryMultiplier>();
}

template<typename T,template<class> class E,template<class> class MVM>
void perfTest(){
    //1 million words
    size_t numberOfWords=1000000;
    size_t numberOfOther=1000;
    size_t dimensionOfWordEmbedding=50;
    size_t dimensionOfOtherEmbedding=20;
    size_t contextLength=1;
    size_t hiddenLayer0NumberOfOutputNodes=60;
    size_t hiddenLayer1NumberOfOutputNodes=18;
    string modelFile="model.faked.bin";
    //set up the model
    perfTestWithBigFakedModelSetup<T>(modelFile,numberOfWords,numberOfOther,dimensionOfWordEmbedding,contextLength,dimensionOfOtherEmbedding,hiddenLayer0NumberOfOutputNodes,hiddenLayer1NumberOfOutputNodes);
    perfTestWithBigFakedModel<T,E,MVM>(modelFile,numberOfWords,numberOfOther);
}

int main( int argc, const char* argv[])
{
    string option;
    if(argc==2){
        option=argv[1];
    }
    if(option=="perf"){
        //default embedding and default matrix vector multiplier
        perfTest<float,EmbeddingWithRawValues,MatrixVectoryMultiplier>();
        perfTest<double,EmbeddingWithRawValues,MatrixVectoryMultiplier>();
    }
    if(option=="perfReal"){
        perfTestWithRealModel<EmbeddingWithRawValues,MatrixVectoryMultiplierBaseline>();
        perfTestWithRealModel<EmbeddingWithRawValues,MatrixVectoryMultiplierMoreUnRolling>();
        perfTestWithRealModel<EmbeddingWithRawValues,MatrixVectoryMultiplier>();
    }
    else if(option=="all"){
        unitTest<float>();
        unitTest<double>();
        perfTest<float,EmbeddingWithRawValues,MatrixVectoryMultiplier>();
        perfTest<double,EmbeddingWithRawValues,MatrixVectoryMultiplier>();
        perfTestWithRealModel<EmbeddingWithRawValues,MatrixVectoryMultiplierBaseline>();
        perfTestWithRealModel<EmbeddingWithRawValues,MatrixVectoryMultiplierMoreUnRolling>();
        perfTestWithRealModel<EmbeddingWithRawValues,MatrixVectoryMultiplier>();
    } else{
        //default: unit test
        unitTest<float>();
        unitTest<double>();
    }
}
