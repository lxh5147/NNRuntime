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

namespace nn {
    using namespace std;
    //Defines an macro that logs error message and stops the current program if some condition does not hold.
    #define ASSERT(condition, message) \
        do { \
            if (! (condition)) { \
                cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                          << " line " << __LINE__ << ": " << message << endl; \
                exit(EXIT_FAILURE); \
            } \
        } while (false)

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
            static void append(const Vector<T>& first,  Vector<T>& second){
                if(first.m_size==0){
                    return;
                }
                //Move to the end of the internal elemeent buffer
                T* buffer=second.m_data.get()+second.m_size;
                memcpy (buffer, first.m_data.get(), sizeof(T)*first.m_size);
                second.m_size += first.m_size;
            }
        public:
            Vector(const shared_ptr<T>& data, const size_t size): m_data(data),m_size(size){}
        private:
            shared_ptr<T> m_data;
            size_t m_size;
    };

    //Defines a matrix of shape row*col.
    template <class T>
    class Matrix {
        public:
            //Returns the column vector of M*v. v.size must equals to the col of this matrix.
            Vector<T> multiply (const Vector<T>& vector) const {
                ASSERT(m_col==vector.size(),"v");
                T* inputElements=vector.data().get();
                T* outputElements=new T[m_row];
                T* matrix=m_data.get();
                for(size_t i=0;i<m_row;++i){
                    outputElements[i]=0;
                    T* matrixElements=matrix+i*m_col;
                    for(size_t j=0;j<m_col;++j){
                        outputElements[i]+=inputElements[j]*matrixElements[j];
                    }
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
            Matrix(const shared_ptr<T>& data, const size_t row, const size_t col): m_data(data), m_row(row), m_col(col) {}
        private:
            const shared_ptr<T> m_data;
            const size_t m_row;
            const size_t m_col;
    };

    //Defines a layer of neural network.
    template<class T, class I>
    class Layer {
        public:
            //Calculates output. 
            virtual Vector<T> calc(const I& input) const = 0;
    };

    //Defines a fully connected hidden layer.
    template<class T>
    class HiddenLayer: public Layer<T, Vector<T>> {
        public:
            //W: output*input matrix
            //b: bias vector with size being the number of output nodes
            HiddenLayer(const Matrix<T>& weights, const Vector<T>& bias, const function<T(const T&)>&  activationFunc): m_weights(weights), m_bias(bias),m_activationFunc(activationFunc){
                ASSERT(weights.row() == bias.size(),"weights and bias");
            }
        public:
            virtual Vector<T> calc(const  Vector<T>& input) const {
                Vector<T> output = m_weights.multiply(input);
                output.plus(m_bias);
                output.apply(m_activationFunc);
                return output;
            }
        private:
            const Matrix<T>& m_weights;
            const Vector<T>& m_bias;
            const function<T(const T&)>& m_activationFunc;
    };

    //Defines iterator interface.        
    template<class T>
    class Iterator{
        public:
            //If next element is available, it will return true and load next element into buffer; otherwise return false.
            virtual bool next(T&)=0;
    };

    //Defines pooling interface.
    template<class T>
    class Pooling {
        public:
            //Calculates the pooled value. It uses a buffer to hold the pooled value, and another buffer to hold the value read from iterator.
            virtual const void calc(T&,Iterator<T>&,T&) const=0;
    };

    //Defines input for neural network.
    template<class T>
    class Input {
        public:
            virtual Vector<T> get() const =0;
            virtual size_t  size() const =0;
        protected:
            Input(const Matrix<T>& embedding): m_embedding(embedding){}
        protected:
            const Matrix<T>& m_embedding;
    };

    //Id of the symbol for padding
    const size_t PADDING_ID = 1;
    //Id of out-of-vocabulary symbol
    const size_t UNK_ID = 0;

    //Defines pooling strategies.
    template<class T>
    class Poolings {
        public:
            enum:int{AVG=0,SUM=1,MAX=2};
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
            static const Pooling<T>&  avg(){
                static  AveragePooling instance;
                return instance;
            } 
            //Returns global sum pooling strategy.
            static const Pooling<T>&  sum(){
                static  SumPooling instance;
                return instance;
            }
            //Returns global max pooling strategy.
            static const Pooling<T>&  max(){
                static  MaxPooling instance;
                return instance;
            }       
        private:  
            //assume plus method is defined by T.
            static void plus(T& target,const T& other){
                target.plus(other);
            }  
            //assume divide method is defined by T.
            template<class U>          
            static void divide(T& target, U denominator){
                target.divide(denominator);
            }
            //assume max method is defined by T.
            static void max(T& target,const T& other){
                target.max(other);
            } 
            //Average pooling strategy.
            class AveragePooling: public Pooling<T> {
                public:
                    virtual const void calc(T& output,Iterator<T>& it,T& buffer) const { 
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
                    virtual const void calc(T& output,Iterator<T>& it,T& buffer) const {
                        bool hasElement = it.next(output);
                        ASSERT(hasElement,"it");
                        while(it.next(buffer)){
                            plus(output,buffer);
                        }
                    }
            };
            //Max pooling strategy.
            class MaxPooling: public Pooling<T> {
                public:
                    virtual const void calc(T& output,Iterator<T>& it,T& buffer) const {  
                        bool hasElement = it.next(output);
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
                size_t dimension = size();
                Vector<T> buffer(shared_ptr<T>(new T[dimension]),dimension);
                Vector<T> output(shared_ptr<T>(new T[dimension]),dimension);
                InputVectorIterator it(*this);
                m_pooling.calc(output,it,buffer);  
                return output;
            }
            //Returns the dimension of the input vector associated with this sequence input.
            virtual size_t  size() const {
                return  Input<T>::m_embedding.col() * (2*m_contextLength + 1 );
            }
        public:
            SequenceInput(const vector<size_t>& idSequence, const size_t contextLength, const Matrix<T>& embedding, const Pooling<Vector<T>>& pooling):
                Input<T>(embedding), m_idSequence(idSequence), m_contextLength(contextLength),m_pooling(pooling){
                ASSERT(m_idSequence.size()>0,"idSequence");
                ASSERT(contextLength>=0,"contextLength");
                for(auto &id:idSequence){
                    ASSERT(id < embedding.row(),"id");
                }
            }
        private:
            //Generates concatenated vector for the window with pos in the middle.
            void generateConcatenatedVector(Vector<T>& concatenationBuffer, size_t pos) const {
                T* embeddingBuffer=Input<T>::m_embedding.data().get();
                T* buffer=concatenationBuffer.data().get();
                size_t col=Input<T>::m_embedding.col();
                size_t size=m_idSequence.size();
                //note: size_t is unsigned int
                int start=(int)pos-(int)m_contextLength;
                int end=pos+m_contextLength;
                for(int i=start;i<=end;++i) {
                    size_t id=i>=0 && i < size ? m_idSequence[i]:PADDING_ID;
                    memcpy(buffer, embeddingBuffer+id*col, sizeof(T)*col);
                    buffer+= col;
                }
            }
        private:
            const vector<size_t> m_idSequence;
            const size_t m_contextLength;
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
                size_t size=Input<T>::m_embedding.col();
                T* buffer=new T[size];              
                T* embedding=Input<T>::m_embedding.data().get();
                memcpy(buffer, embedding+m_id*size, sizeof(T)*size);
                return Vector<T> ( shared_ptr<T>(buffer),size);
            }
            virtual size_t  size() const {
                return  Input<T>::m_embedding.col();
            }
        public:
            NonSequenceInput(const size_t id, const Matrix<T>& embedding):Input<T>(embedding), m_id(id){
                ASSERT(id < embedding.row(),"id");
            }
        private:
            size_t m_id;
    };

    //Defins the input layer of neural network, which consists of a set of sequence/non-sequence inputs.
    template<class T>
    class InputLayer: public Layer<T, vector<reference_wrapper<Input<T>>>>{
        public:
            //Returns the input vector calculated based on sequence/non-sequence inputs.
            virtual Vector<T> calc(const  vector<reference_wrapper<Input<T>>>& inputs) const {
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
            virtual Vector<T> calc(const I& inputs) const = 0;
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
            shared_ptr<InputLayer<T>> m_inputLayer;
            vector<shared_ptr<Layer<T, Vector<T>>>> m_layers;
    };

    //Defines common activation functions.
    template <class T>
    class ActivationFunctions{
        public:
            enum:int {TANH=0,RELU=1};
            //Gets the well-known activation functions with its id.
            static const function<T(const T&)>& get(int id ){
                ASSERT(id==TANH||id==RELU,"id");
                if(id==TANH){
                    return Tanh();
                }
                return ReLU();
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
        private:
            static T _Tanh(const T& t){
                return tanh(t);
            }
            static T _ReLU(const T& t){
                return t>0?t:0;
            }
    }; 

    //Defines NN runtime factory interface. There is 1:1 mapping between NN runtime and its factory.
    template <class T, class I>
    class NNModel {
        public:
            //Loads model from a binary file.
            virtual void load(const char* modelPath)=0;
            //Saves this model to a binary file.
            virtual void save(const char* modelPath) const=0;
            //Predicts with ids as inputs.
            virtual Vector<T> predict(const vector<vector<size_t>>& idsInputs) const=0;
    };
    
    //Defines input information.
    template<class T>
    class InputInfo{
        public:
            enum:int{SEQUENCE_INPUT=0,NON_SEQUENCE_INPUT=1};
        public:
            shared_ptr<Input<T>> getInput(const vector<size_t>& ids) const{
                ASSERT(ids.size()>=1,"ids");
                ASSERT(ids.size()==1||m_inputType==SEQUENCE_INPUT,"ids");
                if(m_inputType==SEQUENCE_INPUT){
                    return shared_ptr<Input<T>>(new SequenceInput<T>(ids, m_embedding,m_contextLength,Poolings<Vector<T>>::get(m_poolingId)));
                }
                else{
                    return shared_ptr<Input<T>>(new NonSequenceInput<T>(ids[0],m_embedding));
                }
            }
        public:
            InputInfo(int inputType,const Matrix<T>& embedding,size_t contextLength,int poolingId):m_inputType(inputType),m_embedding(embedding),m_contextLength(contextLength),m_poolingId(poolingId){
                ASSERT(inputType==SEQUENCE_INPUT||inputType==NON_SEQUENCE_INPUT,"inputType");
                ASSERT(embedding.row()>=2,"embedding");
            }
            InputInfo(const Matrix<T>& embedding):m_inputType(NON_SEQUENCE_INPUT),m_embedding(embedding),m_contextLength(0),m_poolingId(-1){               
                ASSERT(embedding.row()>=2,"embedding");
            }            
        private:
            const int m_inputType;
            const Matrix<T>& m_embedding;
            const size_t m_contextLength;
            const int m_poolingId;
    };
    
    //Defines MLP model.
    template<class T>
    class MLPModel: public NNModel<T,vector<reference_wrapper<Input<T>>>>{
        public:          
            virtual Vector<T> predict(const vector<vector<size_t>>& idsInputs) const{
                vector<shared_ptr<Input<T>>> pInputs=createInputs(idsInputs);
                vector<reference_wrapper<Input<T>>> inputs;
                for(auto &pInput:pInputs){
                    inputs.push_back(reference_wrapper<Input<T>>(*pInput));
                }
                return m_pRuntime->calc(inputs);
            }
            virtual void load(const char* modelPath){
                ASSERT(modelPath,"modelPath");
                ifstream is(modelPath,ios::binary);
                ASSERT(is.is_open(),"is");
                cleanIfNeeded();
                loadInputsInfo(is);
                loadHiddenLayers(is);
                is.close();
                initRuntime();
            }
            virtual void save(const char* modelPath) const{
                ASSERT(modelPath,"modelPath");
                ofstream os (modelPath, ios::binary);
                ASSERT(os.is_open(),"os");
                saveInputsInfo(os);
                saveHiddenLayers(os);
            }
        public:
            MLPModel():m_pRuntime(nullptr){}
            MLPModel(const vector<shared_ptr<InputInfo<T>>>& inputsInfo,const vector<shared_ptr<Matrix<T>>>& embeddings,const vector<shared_ptr<Matrix<T>>>& weights,const vector<shared_ptr<Vector<T>>>& biasVectors,const vector<size_t> activationFunctionIds):m_inputsInfo(inputsInfo),m_embeddings(embeddings),m_weights(weights),m_biasVectors(biasVectors),m_activationFunctionIds(activationFunctionIds){
                initRuntime();
            }
            ~MLPModel(){
                if(m_pRuntime){
                    delete m_pRuntime;
                    m_pRuntime=nullptr;
                }
            }
        private:
            void initRuntime(){
                vector<shared_ptr<Layer<T, Vector<T>>>> layers;
                size_t total=m_weights.size();
                for(size_t i=0;i<total;++i){
                    layers.push_back(shared_ptr<Layer<T, Vector<T>>>(new HiddenLayer<T>(*m_weights[i],*m_biasVectors[i],ActivationFunctions<T>::get(m_activationFunctionIds[i]))));
                }
                layers.push_back(shared_ptr<Layer<T, Vector<T>>>(new SoftmaxLayer<T>()));
                m_pRuntime=new MLP<T>(shared_ptr<InputLayer<T>> (new InputLayer<T>()),layers);
                ASSERT(m_pRuntime,"m_pRuntime");                
            }
            void cleanIfNeeded(){
                 if(m_pRuntime==nullptr){
                    return;
                 }
                delete m_pRuntime;
                m_pRuntime=nullptr;
                m_inputsInfo.clear();
                m_embeddings.clear();
                m_weights.clear();
                m_biasVectors.clear();
                m_activationFunctionIds.clear();
            }
            void loadInputsInfo(istream& is){
                int total=0;
                is>>total;
                int inputType;
                size_t contextLength;
                int poolingId;
                Matrix<T>* pMatrix;
                for(int i=0;i<total;++i){
                    is>>inputType;
                    pMatrix=loadMatrix(is);
                    m_embeddings.push_back(shared_ptr<Matrix<T>>(pMatrix));
                    is>>contextLength>>poolingId;
                    m_inputsInfo.push_back(shared_ptr<InputInfo<T>>(new InputInfo<T>(inputType,*pMatrix,contextLength,poolingId)));
                }
            }
            void loadHiddenLayers(istream& is){
                int total=0;
                is>>total;  
                Matrix<T>* pPreMatrix=nullptr;
                size_t activationFunctionId;
                Matrix<T>* pMatrix;
                Vector<T>* pBias;              
                for(int i=0;i<total;++i){
                    pMatrix=loadMatrix(is);
                    if(pPreMatrix!=nullptr){
                        ASSERT(pPreMatrix->row()==pMatrix->col(),"input vector size");
                    }
                    pPreMatrix=pMatrix;
                    pBias=loadVector(is);
                    is>>activationFunctionId;
                    m_weights.push_back(shared_ptr<Matrix<T>>(pMatrix));
                    m_biasVectors.push_back(shared_ptr<Vector<T>>(pBias));
                    m_activationFunctionIds.push_back(activationFunctionId);
                }
            }
            Matrix<T>* loadMatrix(istream& is) const{
                size_t row ;
                size_t col;
                is>>row>>col;
                T* buffer=new T[row*col];
                is.read(buffer,sizeof(T)*row*col);
                return new Matrix<T>(shared_ptr<T>(buffer),row,col);
            }
            Vector<T>* loadVector(istream& is) const{
                size_t size;
                is>>size;
                T* buffer=new T[size];
                is.read(buffer,sizeof(T)*size);
                return new Vector<T>(shared_ptr<T>(buffer),size);
            }
            void saveInputsInfo(ostream& os) const{
                int total=0;
                os<<total;
                for(auto& pInputInfo:m_inputsInfo){
                    os<<pInputInfo->m_inputType;
                    saveMatrix(os,pInputInfo->m_embedding);
                    os<<pInputInfo->m_contextLength>>pInputInfo->m_poolingId;
                }
            }
            void saveHiddenLayers(ostream& os) const{
                int total=0;
                os<<total;
                for(int i=0;i<total;++i){
                    saveMatrix(os,*m_weights[i]);
                    saveVector(os,*m_biasVectors[i]);
                    os<<m_activationFunctionIds[i];
                }
            }
            void saveMatrix(ostream& os, const Matrix<T>& matrix) const{
                size_t row=matrix.row();
                size_t col=matrix.col();
                os<<row<<col;
                T* buffer=matrix.data().get();
                os.write(buffer,sizeof(T)*row*col);
            }
            Vector<T>* saveVector(ostream& os, const Vector<T>& vector) const{
                size_t size=vector.size();
                os<<size;
                T* buffer=vector.data().get();
                os.write(buffer,sizeof(T)*size);
            }
            //Creates inputs for the runtime
            vector<shared_ptr<Input<T>>> createInputs(const vector<vector<size_t>>& idsInputs) const{
                ASSERT(idsInputs.size()==m_inputsInfo.size(),"idsInputs");
                vector<shared_ptr<Input<T>>> inputs;
                for(size_t i=0;i<idsInputs.size();++i){
                    inputs.push_back(m_inputsInfo[i].getInput(idsInputs[i]));
                }
                return inputs;
            } 
        private:
            NN<T,vector<reference_wrapper<Input<T>>>> *m_pRuntime;
            vector<shared_ptr<InputInfo<T>>> m_inputsInfo; 
            vector<shared_ptr<Matrix<T>>> m_embeddings;
            vector<shared_ptr<Matrix<T>>> m_weights;
            vector<size_t> m_activationFunctionIds;
            vector<shared_ptr<Vector<T>>> m_biasVectors;  
    };
}

#endif
