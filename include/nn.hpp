/*
This file defines the runtime of a neural network.
*/
#ifndef __NN__
#define __NN__
#include <functional>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
namespace nn {
    //Defines an macro that logs error message and stops the current program if some condition does not hold.
    #define ASSERT(condition, message) \
        do { \
            if (! (condition)) { \
                std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                          << " line " << __LINE__ << ": " << message << std::endl; \
                std::exit(EXIT_FAILURE); \
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
            
            void divide(T denominator){
                ASSERT(denominator!=0,"denominator");
                T* elements=m_data.get();
                for(size_t i=0;i<m_size;++i){
                    elements[i]/=denominator;
                }                
            }

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
            void apply(const std::function<T(const T&)>& func){               
                T* elements=m_data.get();
                for(size_t i=0;i<m_size;++i){
                    elements[i]=func(elements[i]);
                }
            }
            //Aggegrates all elements of this vector into one element with an initial value and a binary function.
     	    T aggregate(const std::function<T(const T&,const T&)>& func, const T& initialValue) const {               
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
            const std::shared_ptr<T>& data() const {
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
            Vector(const std::shared_ptr<T>& data, const size_t size): m_data(data),m_size(size){}            
        private:
            std::shared_ptr<T> m_data;
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
                return Vector<T> (std::shared_ptr<T>(outputElements), m_row);            
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
            const std::shared_ptr<T>& data() const {
                return m_data;
            }

        public:
            Matrix(const std::shared_ptr<T>& data, const size_t row, const size_t col): m_data(data), m_row(row), m_col(col) {}
            
        private:
	        const std::shared_ptr<T> m_data;
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
            HiddenLayer(const Matrix<T>& weights, const Vector<T>& bias, std::function<T(const T&)>&  activationFunc): m_weights(weights), m_bias(bias),m_activationFunc(activationFunc){
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
            const Matrix<T> m_weights;
            const Vector<T> m_bias;
            const std::function<T(const T&)> m_activationFunc;
    };
    
    //Defines embedding list.
    template<class T>
    class Embeddings {
        public:
            const Matrix<T>& get(size_t i) const {
                ASSERT(i<m_embeddings.size(),"i");
                return m_embeddings[i];
            }
            
        public:
            Embeddings(const std::vector<Matrix<T>>& embeddings): m_embeddings(embeddings){}           
            
        private:
            const std::vector<Matrix<T>> m_embeddings;     
    };
            
    template<class T>
    class Itorator{
        public:           
            virtual bool next(T&)=0;
    };
    
    template<class T>
    class Pooling {
        public:
            virtual const void calc(T&,Itorator<T>&,T&) const=0;
    };
    
    template<class T>
    void plus(T& target,const T& other){
        target.plus(other);
    }
    
    template<class T,class U>
    void divide(T& target, U denominator){
        target.divide(denominator);
    }

    template<class T>
    void max(T& target,const T& other){
        target.max(other);
    }
    
    template<class T>
    class AveragePooling: public Pooling<T> {
        public:
            virtual const void calc(T& output,Itorator<T>& it,T& buffer) const { 
                bool hasElement = it.next(output);
                ASSERT(hasElement,"it");               
                size_t total=1;                
                while(it.next(buffer)){
                    ++total;
                    plus<T>(output,buffer);
                }             
                divide<T,size_t>(output,total);               
            }
    };
    
    template<class T>
    class SumPooling: public Pooling<T> {
        public:
            virtual const void calc(T& output,Itorator<T>& it,T& buffer) const {
                bool hasElement = it.next(output);
                ASSERT(hasElement,"it");                
                while(it.next(buffer)){                 
                    plus<T>(output,buffer);
                }              
            }
    }; 
    
    template<class T>
    class MaxPooling: public Pooling<T> {
        public:
            virtual const void calc(T& output,Itorator<T>& it,T& buffer) const {  
                bool hasElement = it.next(output);
                ASSERT(hasElement,"it");   
                while(it.next(buffer)){                 
                    max<T>(output,buffer);
                }  
            }
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
            Matrix<T> m_embedding;
    };
    
    //Id of the symbol for padding
    const size_t PADDING_ID = 1;
    //Id of out-of-vocabulary symbol
    const size_t UNK_ID = 0;
    
    //Defines a sequence input. A sequence input has an embedding table, a text window size and a sequence of symbol ids. An embedding table is a matrix, with its i^th row reresenting the embedding of the i^th symbol.
    template<class T>
    class SequenceInput: public Input<T> {
        public:
            virtual Vector<T> get() const {
                size_t dimension = size();             
                Vector<T> buffer(std::shared_ptr<T>(new T[dimension]),dimension);                
                Vector<T> output(std::shared_ptr<T>(new T[dimension]),dimension);
                InputVectorIterator it(*this);
                m_pooling.calc(output,it,buffer);  
                return output;                             
            }
            
            //Returns the dimension of the input vector associated with this sequence input.
            virtual size_t  size() const {
                return  Input<T>::m_embedding.col() * (2*m_contextLength + 1 );
            }

        public:
            SequenceInput(const std::vector<size_t>& idSequence, const size_t contextLength, const Matrix<T>& embedding, const Pooling<Vector<T>>& pooling):
                Input<T>(embedding), m_idSequence(idSequence), m_contextLength(contextLength),m_pooling(pooling){
                ASSERT(m_idSequence.size()>0,"idSequence");
                ASSERT(contextLength>=0,"contextLength");
                for(auto &id:idSequence){
                    ASSERT(id < embedding.row(),"id");
                }
            }           

        private:
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
	        const std::vector<size_t> m_idSequence;
            const size_t m_contextLength;
            const Pooling<Vector<T>>& m_pooling;
            
        private:
            class InputVectorIterator: public Itorator<Vector<T>>{
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
                return Vector<T> ( std::shared_ptr<T>(buffer),size);
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
    class InputLayer: public Layer<T, std::vector<std::reference_wrapper<Input<T>>>>{
        public:
            //Returns the input vector calculated based on sequence/non-sequence inputs.
            virtual Vector<T> calc(const  std::vector<std::reference_wrapper<Input<T>>>& inputs) const {                
                //dimension of the input vector
                size_t size=0;
                for (auto &input:inputs){
                    size+=input.get().size();
                }
                //the output vector has enough element buffer but zero length
                std::shared_ptr<T> data(new T[size]);
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
                return Vector<T>(std::shared_ptr<T>(outputElements),size);
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
    class MLP: public NN<T, std::vector<std::reference_wrapper<Input<T>>>> {
        public:
            virtual Vector<T> calc(const std::vector<std::reference_wrapper<Input<T>>>& inputs) const {               
                Vector<T> output = m_inputLayer.get()->calc(inputs);
                for(auto &layer: m_layers){
                    output=layer.get()->calc(output);
                }
                return output;
            }

        public:
            MLP(const std::shared_ptr<InputLayer<T>>& inputLayer, const std::vector<std::shared_ptr<Layer<T, Vector<T>>>>& layers):m_inputLayer(inputLayer), m_layers(layers){}   
                        
        private:
            std::shared_ptr<InputLayer<T>> m_inputLayer;
            std::vector<std::shared_ptr<Layer<T, Vector<T>>>> m_layers;
    };
    
    //Defines common activation functions
    template <class T>
    struct ActivationFunctions{
        static T Tanh(const T& t){
            return tanh(t);
        }
        
        static T ReLU(const T& t){
            return t>0?t:0;
        }
    }; 
}

#endif
