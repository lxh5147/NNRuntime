/*
This file defines the runtime of a neural network.
*/
#ifndef __NN__
#define __NN__

#include <functional>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <string>
#include <map>
#include <mutex>
#include "md5.hpp"
#include "common.hpp"
#include "quantization.hpp"

namespace nn {
    using namespace std;
    using namespace common;
    using namespace quantization;

    //Defines a vector. A vector consists of an array of elements and their size.
    template <class T>
    class Vector {
        public:
            //Performs element-wise add of this vector and an input vector. This vector is updated to hold the accumulated results.
            void plus(const Vector<T>& vector){
                ASSERT(vector.m_size==m_size,"vector");
                T* elements=m_data.get();
                const T* otherElements=vector.m_data.get();
                for(size_t i=0;i<m_size;++i){
                    elements[i]+=otherElements[i];
                }
            }
            //Performs element-wise divide. This vector is updated to hold the updated results.
            void divide(T denominator){
                ASSERT(denominator!=0,"denominator");
                T* elements=m_data.get();
                for(size_t i=0;i<m_size;++i){
                    elements[i]/=denominator;
                }
            }
            //Performs element-wise max. This vector is updated to hold the updated results.
            void max(const Vector<T>& vector){
                ASSERT(vector.m_size==m_size,"vector");
                T* elements=m_data.get();
                const T* otherElements=vector.m_data.get();
                for(size_t i=0;i<m_size;++i){
                    if(elements[i]<otherElements[i]){
                        elements[i]=otherElements[i];
                    }
                }
            }
            //Applys the unary function to each element of this vector. This vector is updated to reflect the application of the function.
            void apply(const function<T(const T&)>& func){
                T* elements=m_data.get();
                for(size_t i=0;i<m_size;++i){
                    elements[i]=func(elements[i]);
                }
            }
            //Aggegrates all elements of this vector into one element with an initial value and a binary function.
            T aggregate(const function<T(const T&,const T&)>& func, const T& initialValue) const {
                T aggregated=initialValue;
                T* elements=m_data.get();
                for(size_t i=0;i<m_size;++i){
                    aggregated=func(aggregated,elements[i]);
                }
                return aggregated;
            }
            //Returns the number of elements in this vector.
            size_t size() const {
                return m_size;
            }
            //Returns shared pointer that points to the internal elements of this vector.
            const shared_ptr<T>& data() const {
                return m_data;
            }
        public:
            //Appends the first vector to the second vector. The second vector must have enough internal element buffer to hold all elements of the first vector.
            static void append(const Vector<T>& first, Vector<T>& second){
                if(first.m_size==0){
                    return;
                }
                //Move to the end of the internal elemeent buffer
                T* buffer=second.m_data.get()+second.m_size;
                memcpy (buffer, first.m_data.get(), sizeof(T)*first.m_size);
                second.m_size += first.m_size;
            }
        public:
            Vector(shared_ptr<T> data, size_t size): m_data(data),m_size(size){}
        private:
            shared_ptr<T> m_data;
            size_t m_size;
    };

    //Defines helper function to create shared pointer of vector.
    template<typename T>
    shared_ptr<Vector<T>> newVector(T* data, size_t size){
        ASSERT(data,"data");
        return make_shared_ptr(new Vector<T>( make_shared_ptr(data),size));
    }

    //Defines a matrix of shape row*col.
    template <class T>
    class Matrix {
        public:
            //Returns the column vector of M*v. v.size must equals to the col of this matrix.
            Vector<T> multiply (const Vector<T>& vector) const {
                ASSERT(m_col==vector.size(),"v");
                T* inputElements=vector.data().get();
                T* outputElements=new T[m_row]();
                T* outputElement=outputElements;
                T* inputElement;
                T* matrix=m_data.get();
                for(size_t i=0;i<m_row;++i){
                    inputElement=inputElements;
                    for(size_t j=0;j<m_col;++j){
                        *outputElement+=(*inputElement++)*(*matrix++);
                    }
                    ++outputElement;
                }
                return Vector<T> (shared_ptr<T>(outputElements), m_row);
            }
            //Returns the number of rows of this matrix.
            size_t row() const {
                return m_row;
            }
            //Returns the number of columns of this matrix.
            size_t col() const {
                return m_col;
            }
            //Returns the internal element buffer of this matrix.
            const shared_ptr<T>& data() const {
                return m_data;
            }
        public:
            Matrix(shared_ptr<T> data, size_t row, size_t col): m_data(data), m_row(row), m_col(col) {}
        private:
            const shared_ptr<T> m_data;
            const size_t m_row;
            const size_t m_col;
    };

    //Defines helper function to create shared pointer of matrix.
    template<typename T>
    shared_ptr<Matrix<T>> newMatrix(T* data, size_t row, size_t col){
        ASSERT(data,"data");
        return make_shared_ptr(new Matrix<T>(make_shared_ptr(data),row,col));
    }

    //Defines a layer of neural network.
    template<class T, class I>
    class Layer {
        public:
            //Calculates output. 
            virtual Vector<T> calc(const I& input) const = 0;
        public:
            virtual ~Layer(){}
    };

    //Defines a fully connected hidden layer.
    template<class T>
    class HiddenLayer: public Layer<T, Vector<T>> {
        public:
            //W: output*input matrix
            //b: bias vector with size being the number of output nodes
            HiddenLayer(const Matrix<T>& weights, const Vector<T>& bias, const function<T(const T&)>& activationFunc): m_weights(weights), m_bias(bias),m_activationFunc(activationFunc){
                ASSERT(weights.row()==bias.size(),"weights and bias");
            }
        public:
            virtual Vector<T> calc(const  Vector<T>& input) const {
                Vector<T> output = m_weights.multiply(input);
                output.plus(m_bias);
                output.apply(m_activationFunc);
                return output;
            }
        private:
            //hold a copy of weights and bias vector
            const Matrix<T> m_weights;
            const Vector<T> m_bias;
            const function<T(const T&)> m_activationFunc;
    };

    //Defines pooling interface.
    template<class T>
    class Pooling {
        public:
            //Calculates the pooled value. It uses a buffer to hold the pooled value, and another buffer to hold the value read from iterator.
            virtual const void calc(T&, Iterator<T>&, T&) const=0;
    };

    //Defines embedding interface.
    template<class T>
    class Embedding{
        public:
            //Fills the embedding vector into the buffer which should be prepared by the caller.
            virtual void get(size_t, T*) const=0;
            //Returns the raw embedding values if applicable.
            virtual shared_ptr<T> get() const {
                return nullptr;
            }
            //Returns the dimension of embedding vector.
            size_t dimension() const{
                return m_dimension;
            }
            //Returns the number of embedding vectors.
            size_t count() const{
                return m_count;
            }
        public:
            virtual ~Embedding(){}
        protected:
            Embedding(size_t count, size_t dimension):m_count(count), m_dimension(dimension){}
        protected:
            const size_t m_count;
            const size_t m_dimension;
    };

    //Defines embedding based on raw values (no quantization).
    template<class T>
    class EmbeddingWithRawValues: public Embedding<T>{
        public:
            void get(size_t id,T* buffer) const {
                ASSERT(buffer,"buffer");
                ASSERT(id>=0&&id<Embedding<T>::m_count,"id");
                memcpy(buffer,m_data.get()+id*Embedding<T>::m_dimension,sizeof(T)*Embedding<T>::m_dimension);
            }
            shared_ptr<T> get() const {
                return m_data;
            }
        public:
            static shared_ptr<Embedding<T>> create(const Matrix<T>& embedding){
                return make_shared_ptr(new EmbeddingWithRawValues<T>(embedding));
            }
        private:
            EmbeddingWithRawValues(const Matrix<T>& embedding):Embedding<T>(embedding.row(),embedding.col()),m_data(embedding.data()){}
        private:
            const shared_ptr<T> m_data;
    };

    //Defines embedding based on quantized values.
    template<class T,class Q>
    class EmbeddingWithQuantizedValues: public Embedding<T>{
        public:
            void get(size_t id,T* buffer) const {
                ASSERT(buffer,"buffer");
                ASSERT(id>=0&&id<Embedding<T>::m_count,"id");
                Q* quantized=m_data.get() + (id*Embedding<T>::m_dimension);
                for(size_t i=0;i<Embedding<T>::m_dimension;++i){
                    buffer[i]=m_quantizer->unquantize(quantized[i]);
                }
            }
        public:
            static shared_ptr<Embedding<T>> create(const Matrix<T>& embedding){
                auto quantizer=createQuantizer(embedding);
                auto count=embedding.row();
                auto dimension=embedding.col();
                auto size=count*dimension;
                auto quantizedEmbedding=make_shared_ptr(new Q[size]);
                auto rawData=embedding.data().get();
                auto quantizedData=quantizedEmbedding.get();
                for(size_t i=0;i<size;++i){
                    quantizedData[i]=quantizer->quantize(rawData[i]);
                }
                return make_shared_ptr(new EmbeddingWithQuantizedValues<T,Q>(quantizedEmbedding,count,dimension,quantizer));
            }
        private:
            class ValueIterator:public Iterator<T>{
                public:
                    virtual bool next(T& buffer) {
                        if(m_pos>=m_size){
                            return false;
                        }
                        buffer=m_values.get()[m_pos];
                        ++m_pos;
                        return true;
                    }
                    void reset(){
                        m_pos=0;
                    }
                public:
                    ValueIterator(shared_ptr<T> values,size_t size):m_values(values),m_size(size),m_pos(0){}
                private:
                    const shared_ptr<T> m_values;
                    const size_t m_size;
                    size_t m_pos;
            };
            static shared_ptr<Quantizer<T,Q>> createQuantizer(const Matrix<T>& embedding){
                auto size=embedding.row()*embedding.col();
                //use linear quantization
                T min,max;
                ValueIterator it(embedding.data(),size);
                calcMinMax(it,min,max);
                it.reset();
                return LinearQuantizer<T,Q>::create(it,MaxValue<Q>(),min,max);
            }
        private:
            EmbeddingWithQuantizedValues(shared_ptr<Q> embedding,size_t count, size_t dimension, shared_ptr<Quantizer<T,Q>> quantizer):Embedding<T>(count,dimension),m_data(embedding),m_quantizer(quantizer) {}
        private:
            const shared_ptr<Q> m_data;
            //Use a pointer so that it can use any type of quantizers
            const shared_ptr<Quantizer<T,Q>> m_quantizer;
    };

    //Defines input for neural network.
    template<class T>
    class Input {
        public:
            virtual Vector<T> get() const =0;
            virtual size_t  size() const =0;
        public:
            virtual ~Input(){}
        protected:
            Input(shared_ptr<Embedding<T>> embedding): m_embedding(embedding){}
        protected:
            //Pointer to any type of embedding
            const shared_ptr<Embedding<T>> m_embedding;
    };

    //Id of the symbol for padding
    const size_t PADDING_ID = 1;
    //Id of out-of-vocabulary symbol
    const size_t UNK_ID = 0;

    //Defines pooling strategies.
    template<class T>
    class Poolings {
        public:
            enum:int{AVG=0, SUM=1, MAX=2};
            //Gets the pre-defined pooling strategy.
            static const Pooling<T>& get(int id){
                ASSERT(id==AVG||id==SUM||id==MAX,"id");
                if(id==AVG){
                    return avg();
                }
                if(id==SUM){
                    return sum();
                }
                return max();
            }
        private:  
            //Returns global avg pooling strategy.
            static const Pooling<T>& avg(){
                //singleton implemented with local static variable
                static  AveragePooling instance;
                return instance;
            } 
            //Returns global sum pooling strategy.
            static const Pooling<T>& sum(){
                static  SumPooling instance;
                return instance;
            }
            //Returns global max pooling strategy.
            static const Pooling<T>& max(){
                static  MaxPooling instance;
                return instance;
            }       
        private:  
            //assume plus method is defined by T.
            static void plus(T& target, const T& other){
                target.plus(other);
            }  
            //assume divide method is defined by T.
            template<class U>          
            static void divide(T& target, U denominator){
                target.divide(denominator);
            }
            //assume max method is defined by T.
            static void max(T& target, const T& other){
                target.max(other);
            } 
            //Average pooling strategy.
            class AveragePooling: public Pooling<T> {
                public:
                    virtual const void calc(T& output,Iterator<T>& it, T& buffer) const { 
                        bool hasElement = it.next(output);
                        ASSERT(hasElement,"it");
                        size_t total=1;
                        while(it.next(buffer)){
                            ++total;
                            plus(output,buffer);
                        }
                        divide<size_t>(output,total);
                    }
            };
            //Sum pooling strategy.
            class SumPooling: public Pooling<T> {
                public:
                    virtual const void calc(T& output,Iterator<T>& it, T& buffer) const {
                        bool hasElement=it.next(output);
                        ASSERT(hasElement,"it");
                        while(it.next(buffer)){
                            plus(output,buffer);
                        }
                    }
            };
            //Max pooling strategy.
            class MaxPooling: public Pooling<T> {
                public:
                    virtual const void calc(T& output, Iterator<T>& it, T& buffer) const {  
                        bool hasElement=it.next(output);
                        ASSERT(hasElement,"it");   
                        while(it.next(buffer)){
                            max(output,buffer);
                        }
                    }
            };
    };

    //Defines a sequence input. A sequence input has an embedding table, a text window size, a sequence of symbol ids and a pooling strategy. An embedding table is a matrix, with its i^th row reresenting the embedding of the i^th symbol.
    template<class T>
    class SequenceInput: public Input<T> {
        public:
            virtual Vector<T> get() const {
                size_t dimension=size();
                Vector<T> buffer(shared_ptr<T>(new T[dimension]),dimension);
                Vector<T> output(shared_ptr<T>(new T[dimension]),dimension);
                InputVectorIterator it(*this);
                m_pooling.calc(output,it,buffer);  
                return output;
            }
            //Returns the dimension of the input vector associated with this sequence input.
            virtual size_t  size() const {
                return  Input<T>::m_embedding->dimension() * (2*m_contextLength + 1 );
            }
        public:
            SequenceInput(const vector<size_t>& idSequence, size_t contextLength, shared_ptr<Embedding<T>> embedding, const Pooling<Vector<T>>& pooling):
                Input<T>(embedding), m_idSequence(idSequence), m_contextLength(contextLength),m_pooling(pooling){
                ASSERT(m_idSequence.size()>0,"idSequence");
                for(auto &id:idSequence){
                    ASSERT(id<embedding->count(),"id");
                }
            }
        private:
            //Generates concatenated vector for the window with pos in the middle.
            void generateConcatenatedVector(Vector<T>& concatenationBuffer, size_t pos) const{
                auto embedding=Input<T>::m_embedding;
                size_t dimension=embedding->dimension();
                T* buffer=concatenationBuffer.data().get();
                size_t size=m_idSequence.size();
                //note: size_t is unsigned int
                int start=(int)pos-(int)m_contextLength;
                int end=pos+m_contextLength;
                for(int i=start;i<=end;++i) {
                    size_t id=i>=0 && (size_t)i < size ? m_idSequence[i]:PADDING_ID;
                    embedding->get(id,buffer);
                    buffer+=dimension;
                }
            }
        private:
            const vector<size_t> m_idSequence;
            const size_t m_contextLength;
            //pointer to a global pooling object.
            const Pooling<Vector<T>>& m_pooling;
        private:
            //Defines input vector iterator. Each input vector is a concatenated vector for a window.
            class InputVectorIterator: public Iterator<Vector<T>>{
                public:
                    virtual bool next(Vector<T>& buffer) {
                        if(m_pos<m_sequenceInput.m_idSequence.size()){
                            m_sequenceInput.generateConcatenatedVector(buffer,m_pos++);
                            return true;
                        }
                        else{
                            return false;
                        }
                    }
                public:
                    InputVectorIterator(const SequenceInput<T>& sequenceInput):m_sequenceInput(sequenceInput),m_pos(0){}
                private:
                    const SequenceInput<T>& m_sequenceInput;
                    size_t m_pos;
            };
    };
    
    //Defines a non-sequence input. A non-sequence input has an embedding table, and a symbol id.
    template<class T>
    class NonSequenceInput: public Input<T> {
        public:
            virtual Vector<T> get() const {
                auto embedding=Input<T>::m_embedding;
                size_t size=embedding->dimension();
                T* buffer=new T[size];
                embedding->get(m_id,buffer);
                return Vector<T> ( shared_ptr<T>(buffer),size);
            }
            virtual size_t size() const {
                return  Input<T>::m_embedding->dimension();
            }
        public:
            NonSequenceInput(size_t id, shared_ptr<Embedding<T>> embedding):Input<T>(embedding), m_id(id){
                ASSERT(id<embedding->count(),"id");
            }
        private:
            const size_t m_id;
    };

    //Defins the input layer of neural network, which consists of a set of sequence/non-sequence inputs.
    //Use reference wrapper to indicate the the input should always exist.
    template<class T>
    class InputLayer: public Layer<T, vector<reference_wrapper<Input<T>>>>{
        public:
            //Returns the input vector calculated based on sequence/non-sequence inputs.
            virtual Vector<T> calc(const vector<reference_wrapper<Input<T>>>& inputs) const {
                //dimension of the input vector
                size_t size=0;
                for (auto &input:inputs){
                    size+=input.get().size();
                }
                //the output vector has enough element buffer but zero length
                shared_ptr<T> data(new T[size]);
                Vector<T> output(data,0);
                for (auto &input:inputs) {
                    //append to the output vector
                    Vector<T>::append(input.get().get(), output);
                }
                return output;
            }
    };

    //Defines the softmax layer.
    template <class T>
    class SoftmaxLayer: public Layer<T, Vector<T>> {
        public:
            virtual Vector<T> calc(const  Vector<T>& input) const {
                ASSERT(input.size() > 0, "input");
                //ref to: http://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
                size_t size=input.size();
                T* inputElements=input.data().get();
                T max=inputElements[0];
                for(size_t i=1;i<size;++i){
                    if(max<inputElements[i]){
                        max=inputElements[i];
                    }
                }
                double expSum=0;
                for(size_t i=0;i<size;++i){
                    expSum+=exp(inputElements[i]-max);
                }
                double logExpSum=max+log(expSum);
                T* outputElements=new T[size];
                for(size_t i=0;i<size;++i){
                    outputElements[i]=exp(inputElements[i]-logExpSum);
                }
                //optimized by compiler so that no temporal object is generated
                return Vector<T>(shared_ptr<T>(outputElements),size);
            }
    };

    //Defines a neural network compuation network.
    template <class T, class I>
    class NN {
        public:
            //Calculates outputs given the inputs. Note that a neurual network foundmentally can be considered as a function defined by input/output spec.
            virtual Vector<T> calc(const I& inputs) const = 0;
        public:
            virtual ~NN(){}
    };

    //Defines a multiple layer neural network, consisting of an input layer and a list of other layers.
    template <class T>
    class MLP: public NN<T, vector<reference_wrapper<Input<T>>>> {
        public:
            virtual Vector<T> calc(const vector<reference_wrapper<Input<T>>>& inputs) const {
                Vector<T> output = m_inputLayer.get()->calc(inputs);
                for(auto &layer: m_layers){
                    output=layer.get()->calc(output);
                }
                return output;
            }
        public:
            MLP(const shared_ptr<InputLayer<T>>& inputLayer,const vector<shared_ptr<Layer<T,Vector<T>>>>& layers):m_inputLayer(inputLayer),m_layers(layers){}
        private:
            const shared_ptr<InputLayer<T>> m_inputLayer;
            const vector<shared_ptr<Layer<T, Vector<T>>>> m_layers;
    };

    //Defines common activation functions.
    template <class T>
    class ActivationFunctions{
        public:
            enum:int {TANH=0, RELU=1, IDENTITY=2};
            //Gets the well-known activation functions with its id.
            static const function<T(const T&)>& get(int id ){
                ASSERT(id==TANH||id==RELU||id==IDENTITY,"id");
                if(id==TANH){
                    return Tanh();
                }
                if(id==RELU){
                    return ReLU();
                }
                return Identity();
            }
        private:
            static const function<T(const T&)>& Tanh(){
                static function<T(const T&)> instance(_Tanh);
                return instance;
            }
            static const function<T(const T&)>& ReLU(){
                static function<T(const T&)> instance(_ReLU);
                return instance;
            }
            static const function<T(const T&)>& Identity(){
                static function<T(const T&)> instance(_Identity);
                return instance;
            }            
        private:
            static T _Tanh(const T& t){
                return tanh(t);
            }
            static T _ReLU(const T& t){
                return t>0?t:0;
            }
            static T _Identity(const T& t){
                return t;
            }
    }; 

    //Defines NN model interface. It can be considered as extension of the nn runtime, which accepts a list of id sequences as input.
    template <class T>
    class NNModel {
        public:
            //Predicts with ids as inputs.
            virtual Vector<T> predict(const vector<vector<size_t>>& idsInputs) const=0;
        public:
            virtual ~NNModel(){}
    };

    //Defines a helper function to create a shared pointer of sequence input.
    template<typename T>
    shared_ptr<Input<T>> newInput(const vector<size_t>& idSequence, size_t contextLength, shared_ptr<Embedding<T>> embedding, int poolingId){
        return make_shared_ptr(new SequenceInput<T>(idSequence,contextLength,embedding,Poolings<Vector<T>>::get(poolingId)));
    }

    //Defines a helper function to create a shared pointer of non-sequence input.
    template<typename T>
    shared_ptr<Input<T>> newInput(size_t id, shared_ptr<Embedding<T>> embedding){
        return make_shared_ptr(new NonSequenceInput<T>(id,embedding));
    }

    //Defines input information.
    template<class T>
    class InputInfo{
        public:
            enum:int{SEQUENCE_INPUT=0, NON_SEQUENCE_INPUT=1};
        public:
            //Returns input vector corresponding to the input id sequence.
            shared_ptr<Input<T>> getInput(const vector<size_t>& ids) const {
                ASSERT(ids.size()>=1,"ids");
                ASSERT(ids.size()==1||m_inputType==SEQUENCE_INPUT,"ids");
                if(m_inputType==SEQUENCE_INPUT){
                    return newInput(ids,m_contextLength,m_embedding,m_poolingId);
                }
                else{
                    return newInput(ids[0],m_embedding);
                }
            }
        public:
            InputInfo(int inputType, shared_ptr<Embedding<T>> embedding, size_t contextLength, int poolingId):m_inputType(inputType),m_embedding(embedding),m_contextLength(contextLength),m_poolingId(poolingId){
                ASSERT(inputType==SEQUENCE_INPUT||inputType==NON_SEQUENCE_INPUT,"inputType");
                ASSERT(embedding!=nullptr && embedding->count()>=2,"embedding");
            }
            InputInfo(shared_ptr<Embedding<T>> embedding, size_t contextLength, int poolingId):InputInfo<T>(SEQUENCE_INPUT,embedding,contextLength,poolingId){}
            InputInfo(shared_ptr<Embedding<T>> embedding):InputInfo<T>(NON_SEQUENCE_INPUT,embedding,0,-1){}
        public:
            const int inputType() const {
                return m_inputType;
            }
            const shared_ptr<Embedding<T>> embedding() const {
                return m_embedding;
            }
            const size_t contextLength() const {
                return m_contextLength;
            } 
            const int poolingId() const {
                return m_poolingId;
            }
        private:
            const int m_inputType;
            const shared_ptr<Embedding<T>> m_embedding;
            const size_t m_contextLength;
            const int m_poolingId;
    };

    //Defines helper function to create shared pointer for InputInfo.
    template<typename T>
    shared_ptr<InputInfo<T>> newInputInfo(int inputType, shared_ptr<Embedding<T>> embedding, size_t contextLength, int poolingId){
        return make_shared_ptr(new InputInfo<T>(inputType,embedding,contextLength,poolingId));
    }

    //Defines helper function to create shared pointer for InputInfo.
    template<typename T>
    shared_ptr<InputInfo<T>> newInputInfo(shared_ptr<Embedding<T>> embedding, size_t contextLength, int poolingId){
        return make_shared_ptr(new InputInfo<T>(embedding,contextLength,poolingId));
    }

    //Defines helper function to create shared pointer for InputInfo.
    template<typename T>
    shared_ptr<InputInfo<T>> newInputInfo(shared_ptr<Embedding<T>> embedding){
        return make_shared_ptr(new InputInfo<T>(embedding));
    }

    //Defines MLP model.
    template<class T>
    class MLPModel: public NNModel<T>{
        public:
            virtual Vector<T> predict(const vector<vector<size_t>>& idsInputs) const {
                vector<shared_ptr<Input<T>>> pInputs=createInputs(idsInputs);
                vector<reference_wrapper<Input<T>>> inputs;
                for(auto &pInput:pInputs){
                    inputs.push_back(reference_wrapper<Input<T>>(*pInput));
                }
                return m_pRuntime->calc(inputs);
            }
        public:
            MLPModel(const vector<shared_ptr<InputInfo<T>>>& inputsInfo, const vector<shared_ptr<Matrix<T>>>& weights, const vector<shared_ptr<Vector<T>>>& biasVectors, const vector<size_t> activationFunctionIds, bool normalizeOutputWithSoftmax=true):m_inputsInfo(inputsInfo){
                ASSERT(weights.size()==biasVectors.size(),"layer");
                ASSERT(weights.size()==activationFunctionIds.size(),"layer");
                m_pRuntime=createRuntime(weights,biasVectors,activationFunctionIds,normalizeOutputWithSoftmax);
            }

        private:
            static shared_ptr<NN<T,vector<reference_wrapper<Input<T>>>>> createRuntime(const vector<shared_ptr<Matrix<T>>>& weights, const vector<shared_ptr<Vector<T>>>& biasVectors, const vector<size_t>& activationFunctionIds, bool normalizeOutputWithSoftmax){
                vector<shared_ptr<Layer<T, Vector<T>>>> layers;
                size_t total=weights.size();
                for(size_t i=0;i<total;++i){
                    //Hidden layer holders a copy of connection weights and bias vectors.
                    layers.push_back(make_shared_ptr(new HiddenLayer<T>(*weights[i],*biasVectors[i],ActivationFunctions<T>::get(activationFunctionIds[i]))));
                }
                if(normalizeOutputWithSoftmax){
                    layers.push_back(make_shared_ptr(new SoftmaxLayer<T>()));
                }
                return make_shared_ptr(new MLP<T>(make_shared_ptr(new InputLayer<T>()),layers));
            }
            //Creates inputs for the runtime
            vector<shared_ptr<Input<T>>> createInputs(const vector<vector<size_t>>& idsInputs) const{
                ASSERT(idsInputs.size()==m_inputsInfo.size(),"idsInputs");
                vector<shared_ptr<Input<T>>> inputs;
                for(size_t i=0;i<idsInputs.size();++i){
                    inputs.push_back(m_inputsInfo[i]->getInput(idsInputs[i]));
                }
                return inputs;
            }
        private:
            shared_ptr<NN<T,vector<reference_wrapper<Input<T>>>>> m_pRuntime;
            const vector<shared_ptr<InputInfo<T>>> m_inputsInfo;
    };

    //Calculates md5 of data in memory
    template<typename T>
    string md5(T* data,size_t size){
        ASSERT(data,"data");
        ASSERT(size>0,"size");
        return md5::MD5().digestMemory(reinterpret_cast<md5::BYTE*>(data), sizeof(T)*size);
    }

    template<class T> using EmbeddingWith16BitsQuantizedValues = EmbeddingWithQuantizedValues<T,unsigned short>;

    //Defines MLP model factory.
    template<class T, template<class> class E=EmbeddingWithRawValues>
    class MLPModelFactory {
        public:
            static void save(const string& modelPath, const vector<shared_ptr<InputInfo<T>>>& inputsInfo, const vector<shared_ptr<Matrix<T>>>& weights, const vector<shared_ptr<Vector<T>>>& biasVectors, const vector<size_t> activationFunctionIds){
                ofstream os(modelPath, ios::binary);
                ASSERT(os.is_open(),"os");
                saveInputsInfo(os,inputsInfo);
                saveHiddenLayers(os,weights,biasVectors,activationFunctionIds);
            }
            static shared_ptr<MLPModel<T>> load(const string& modelPath, bool normalizeOutputWithSoftmax=true){
                vector<shared_ptr<InputInfo<T>>> inputsInfo;
                vector<shared_ptr<Matrix<T>>> weights;
                vector<shared_ptr<Vector<T>>> biasVectors;
                vector<size_t> activationFunctionIds;
                ifstream is(modelPath,ios::binary);
                ASSERT(is.is_open(),"is");
                loadInputsInfo(is,inputsInfo);
                loadHiddenLayers(is,weights,biasVectors,activationFunctionIds);
                is.close();
                return make_shared_ptr(new MLPModel<T>(inputsInfo,weights,biasVectors,activationFunctionIds,normalizeOutputWithSoftmax));
            }
        private:
            static void saveInputsInfo(ostream& os, const vector<shared_ptr<InputInfo<T>>>& inputsInfo){
                save(os,inputsInfo.size());
                for(auto& pInputInfo:inputsInfo){
                    save(os,pInputInfo->inputType());
                    auto pEmbedding=pInputInfo->embedding();
                    //only support raw embedding
                    ASSERT(pEmbedding->get(),"embedding");
                    Matrix<T> matrix(pEmbedding->get(),pEmbedding->count(),pEmbedding->dimension());
                    saveEmbedding(os,matrix);
                    save(os,pInputInfo->contextLength());
                    save(os,pInputInfo->poolingId());
                }
            }
            static void saveHiddenLayers(ostream& os, const vector<shared_ptr<Matrix<T>>>& weights, const vector<shared_ptr<Vector<T>>>& biasVectors, const vector<size_t> activationFunctionIds){
                size_t size=weights.size();
                save(os,size);
                for(size_t i=0;i<size;++i){
                    save(os,*weights[i]);
                    save(os,*biasVectors[i]);
                    save(os,activationFunctionIds[i]);
                }
            }
            static void saveEmbedding(ostream& os, const Matrix<T>& embedding){
                //save md5 to avoid computing it whiling loading this model
                save(os,md5(embedding.data().get(),embedding.row()*embedding.col()));
                save(os,embedding);
            }
            template<typename V>
            static void save(ostream& os, const V& value){
                os.write(reinterpret_cast<const char*>(&value),sizeof(V));
                ASSERT(os,"os");
            }
            static void save(ostream& os, const string& value){
                size_t size=value.size();
                save(os,size);
                os.write(value.c_str(),size);
                ASSERT(os,"os");
            }
            static void save(ostream& os, const Matrix<T>& matrix){
                size_t row=matrix.row();
                size_t col=matrix.col();
                save(os,row);
                save(os,col);
                T* buffer=matrix.data().get();
                os.write(reinterpret_cast<const char*>(buffer),sizeof(T)*row*col);
                ASSERT(os,"os");
            }
            static void save(ostream& os, const Vector<T>& vector){
                size_t size=vector.size();
                save(os,size);
                T* buffer=vector.data().get();
                os.write(reinterpret_cast<const char*>(buffer),sizeof(T)*size);
                ASSERT(os,"os");
            }
            static void loadInputsInfo(istream& is, vector<shared_ptr<InputInfo<T>>>& inputsInfo){
                size_t total=0;
                load(is,total);
                int inputType;
                size_t contextLength;
                int poolingId;
                shared_ptr<Embedding<T>> pEmbedding;
                for(size_t i=0;i<total;++i){
                    load(is,inputType);
                    pEmbedding=loadEmbedding(is);
                    ASSERT(pEmbedding,"pEmbedding");
                    load(is,contextLength);
                    load(is,poolingId);
                    inputsInfo.push_back(newInputInfo(inputType,pEmbedding,contextLength,poolingId));
                }
            }
            static void loadHiddenLayers(istream& is,vector<shared_ptr<Matrix<T>>>& weights,vector<shared_ptr<Vector<T>>>& biasVectors,vector<size_t>& activationFunctionIds){
                size_t total=0;
                load(is,total);
                shared_ptr<Matrix<T>> pPreMatrix=nullptr;
                size_t activationFunctionId;
                shared_ptr<Matrix<T>> pMatrix;
                shared_ptr<Vector<T>> pBias;
                for(size_t i=0;i<total;++i){
                    pMatrix=loadMatrix(is);
                    if(pPreMatrix!=nullptr){
                        ASSERT(pPreMatrix->row()==pMatrix->col(),"input vector size");
                    }
                    pPreMatrix=pMatrix;
                    pBias=loadVector(is);
                    load(is,activationFunctionId);
                    weights.push_back(pMatrix);
                    biasVectors.push_back(pBias);
                    activationFunctionIds.push_back(activationFunctionId);
                }
            }
            static void load(istream& is,size_t& row, size_t& col){
                load(is,row);
                load(is,col);
            }
            static shared_ptr<Matrix<T>> loadMatrix(istream& is){
                size_t row ;
                size_t col;
                load(is,row,col);
                size_t size=row*col;
                T* buffer=new T[size];
                load(is,buffer,size);
                return make_shared_ptr(new Matrix<T>(shared_ptr<T>(buffer),row,col));
            }
            static shared_ptr<Embedding<T>> loadEmbedding(istream& is){
                //singleton global embedding cache implemented by a local static variable
                static Cache<Embedding<T>> cache;
                string md5;
                load(is,md5);
                size_t row ;
                size_t col;
                load(is,row,col);
                size_t size=row*col;
                shared_ptr<Embedding<T>> pEmbedding=cache.get(md5);
                if(pEmbedding==nullptr){
                    auto buffer=make_shared_ptr(new T[size]);
                    load(is,buffer.get(),size);
                    Matrix<T> matrix(buffer,row,col);
                    pEmbedding=E<T>::create(matrix);
                    cache.put(md5,pEmbedding);
                }else{
                    is.ignore(size*sizeof(T));
                }
                return pEmbedding;
            }
            static shared_ptr<Vector<T>> loadVector(istream& is){
                size_t size;
                load(is,size);
                T* buffer=new T[size];
                load(is,buffer,size);
                return make_shared_ptr(new Vector<T>(shared_ptr<T>(buffer),size));
            }
            template<typename V>
            static void load(istream& is, V& value){
                is.read(reinterpret_cast<char*>(&value),sizeof(V));
            }
            static void load(istream& is, string& value){
                size_t size;
                load(is,size);
                value.resize(size);
                is.read(&value[0], size);
            }
            static void load(istream& is, T* buffer, size_t size){
                is.read(reinterpret_cast<char*>(buffer),sizeof(T)*size);
            }
    };

}

#endif
