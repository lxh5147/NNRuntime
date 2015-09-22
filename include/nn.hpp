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
    //Defines an macro that logs error message and stops the current program if some condition does not hold. It only works when NDEBUG is not defined.
    #ifndef NDEBUG
    #   define ASSERT(condition, message) \
        do { \
            if (! (condition)) { \
                std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                          << " line " << __LINE__ << ": " << message << std::endl; \
                std::exit(EXIT_FAILURE); \
            } \
        } while (false)
    #else
    #   define ASSERT(condition, message) do { } while (false)
    #endif   

    //Defines a vector. A vector consists of an array of elements and their size.  
    template <class T>
    class Vector {
        public:
            //Performs element-wise add of this vector and an input vector. This vector is updated to hold the accumulated results.
            void plus(const Vector<T>& v){
                ASSERT(v.m_size==m_size,"v");
                T* a=m_data.get();
                const T* b=v.m_data.get();
                for(size_t i=0;i<m_size;++i){
                    a[i]+=b[i];
                }
            }
            //Applys the unary function to each element of this vector. This vector is updated to reflect the application of the function.
            void apply(const std::function<T(const T&)>& func){               
                T* a=m_data.get();
                for(size_t i=0;i<m_size;++i){
                    a[i]=func(a[i]);
                }
            }
            //Aggegrates all elements of this vector into one element with an initial value and a binary function.
     	    T aggregate(const std::function<T(const T&,const T&)>& func, const T& t0) const {               
	            T t=t0;
                T* a=m_data.get();
                for(size_t i=0;i<m_size;++i){
                    t=func(t,a[i]);
                }
                return t;
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
            static void append(const Vector<T>& v,  Vector<T>& r){
                if(v.m_size==0){
                    return;
                }
                //Move to the end of the internal elemeent buffer
                T* a=r.m_data.get()+r.m_size;
                memcpy (a, v.m_data.get(), sizeof(T)*v.m_size);
                r.m_size += v.m_size;
            }

        public:
            Vector(const std::shared_ptr<T>& data, const size_t size): m_data(data),m_size(size){}
            Vector(const Vector<T>& other): m_data(other.m_data),m_size(other.m_size){}
            Vector(Vector<T>&& other): m_data(std::move(other.m_data)),m_size(other.m_size){}
        private:
            const std::shared_ptr<T> m_data;
            size_t m_size;
    };
    
    //Defines a matrix of shape row*col.
    template <class T>
    class Matrix {
        public:
            //Returns the column vector of M*v. v.size must equals to the col of this matrix.
            Vector<T> multiply (const Vector<T>&  v) const {
                ASSERT(m_col==v.size(),"v");
                T* a=v.data().get();
                T* y=new T[m_row];
                T* M=m_data.get();
                for(size_t i=0;i<m_row;++i){
                    y[i]=0;
                    T* b=M+i*m_col;
                    for(size_t j=0;j<m_col;++j){
                        y[i]+=a[j]*b[j];
                    }
                }              
                return Vector<T> (std::shared_ptr<T>(y), m_row);            
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
            Matrix(const Matrix<T>& other): m_data(other.m_data),m_row(other.m_row),m_col(other.m_col){}
            Matrix(Matrix<T>&& other): m_data(std::move(other.m_data)),m_row(other.m_row),m_col(other.m_col){}
            
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
            HiddenLayer(const Matrix<T>& W, const Vector<T>& b, std::function<T(const T&)>&  activationFunc): m_W(W), m_b(b),m_activationFunc(activationFunc){
                ASSERT(W.row() == b.size(),"W and b");               
            }
            HiddenLayer(const HiddenLayer<T>& other): m_W(other.m_W), m_b(other.m_b),m_activationFunc(other.m_activationFunc){}
            HiddenLayer(HiddenLayer<T>&& other): m_W(std::move(other.m_W)), m_b(std::move(other.m_b)),m_activationFunc(other.m_activationFunc){}           

        public:
            virtual Vector<T> calc(const  Vector<T>& input) const {               
                Vector<T> r = m_W.multiply(input);
                r.plus(m_b);
                r.apply(m_activationFunc);
                return r;
            }

        private:
            Matrix<T> m_W;
            Vector<T> m_b;
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
            Embeddings(const Embeddings<T>& other): m_embeddings(other.m_embeddings){}
            Embeddings(Embeddings<T>&& other): m_embeddings(std::move(other.m_embeddings)){}
            
        private:
            const std::vector<Matrix<T>> m_embeddings;     
    };
    
    typedef unsigned int UINT;
    
    //Defines input for neural network.
    template<class T>
    class Input {
        public:
            virtual Vector<T> get() const = 0;
            virtual size_t  size() const = 0;

        protected:
            Input(const Matrix<T>& embedding): m_embedding(embedding){}
            Input(const Input<T>& other): m_embedding(other.m_embedding){}
            Input(Input<T>&& other): m_embedding(std::move(other.m_embedding)){}
            
        protected:
            const Matrix<T> m_embedding;
    };
    
    //Id of the symbol for padding
    const UINT PADDING_ID = 1;
    //Id of out-of-vocabulary symbol
    const UINT UNK_ID = 0;
    
    //Defines a sequence input. A sequence input has an embedding table, a text window size and a sequence of symbol ids. An embedding table is a matrix, with its i^th row reresenting the embedding of the i^th symbol.
    template<class T>
    class SequenceInput: public Input<T> {
        public:
            virtual Vector<T> get() const {                
                //generate a vectors for each text window
                size_t dimension = size();             
                Vector<T> v(std::shared_ptr<T>(new T[dimension]),dimension);                
                Vector<T> r(std::shared_ptr<T>(new T[dimension]),dimension);
                memset (r.data().get(),0,sizeof(T)*dimension);    
                size_t countOfIds=m_idSequence.size();
                for(UINT pos=0;pos<countOfIds;++pos){
                    generateConcatenatedVector(v,pos);       
                    r.plus(v);       
                }
                //average pooling
                auto divide=[countOfIds](const T& t){return t/countOfIds;}; 
                r.apply(divide);
                return r;
            }
            //Returns the dimension of the input vector associated with this sequence input.
            virtual size_t  size() const {
                return  Input<T>::m_embedding.col() * (2*m_contextLength + 1 );
            }

        public:
            SequenceInput(const std::vector<UINT>& idSequence, const size_t contextLength, const Matrix<T>& embedding):
                Input<T>(embedding), m_idSequence(idSequence), m_contextLength(contextLength){
                ASSERT(m_idSequence.size()>0,"idSequence");
                ASSERT(contextLength>=0,"contextLength");
                for(auto &id:idSequence){
                    ASSERT(id>=0 && id < embedding.row(),"id");
                }
            }
            SequenceInput(const SequenceInput<T>& other): Input<T>(other.m_embedding), m_idSequence(other.m_idSequence), m_contextLength(other.m_contextLength){}
            SequenceInput(SequenceInput<T>&& other): Input<T>(std::move(other.m_embedding)), m_idSequence(std::move(other.m_idSequence)), m_contextLength(other.m_contextLength){}

        private:
            void generateConcatenatedVector(Vector<T>& v, UINT pos) const {                
                T* E=Input<T>::m_embedding.data().get();
                T* c=v.data().get();
                size_t col=Input<T>::m_embedding.col();
                size_t size=m_idSequence.size();
                //note: size_t is unsigned int
                int start=(int)pos-(int)m_contextLength;               
                int end=pos+m_contextLength;                
                for(int i=start;i<=end;++i) {
                    UINT id=i>=0 && i < size ? m_idSequence[i]:PADDING_ID;                  
                    memcpy(c, E+id*col, sizeof(T)*col);
                    c+= col;
                }
            }

        private:
	        const std::vector<UINT> m_idSequence;
            const size_t m_contextLength;
    };
    
    //Defines a non-sequence input. A non-sequence input has an embedding table, and a symbol id.
    template<class T>
    class NonSequenceInput: public Input<T> {
        public:
            virtual Vector<T> get() const {
                size_t size=Input<T>::m_embedding.col();
                T* a=new T[size];
                std::shared_ptr<T> data(a);
                T* E=Input<T>::m_embedding.data().get();
                memcpy(a, E+m_id*size, sizeof(T)*size);
                return Vector<T> (data,size);
            }

            virtual size_t  size() const {
                return  Input<T>::m_embedding.col();
            }

        public:
            NonSequenceInput(const UINT id, const Matrix<T>& embedding):Input<T>(embedding), m_id(id){
                ASSERT(id>=0 && id < embedding.row(),"id");
            }
            NonSequenceInput(const NonSequenceInput<T>& other):Input<T>(other.m_embedding),m_id(other.m_id){}            
            NonSequenceInput(NonSequenceInput<T>&& other):Input<T>(other.m_embedding),m_id(other.m_id){} 
             
        private:
	        const UINT m_id;
    };

    //Defins the input layer of neural network, which consists of a set of sequence/non-sequence inputs.
    template<class T>
    class InputLayer: public Layer<T, std::vector<std::reference_wrapper<Input<T>>>>{
        public:
            //Returns the input vector calculated based on sequence/non-sequence inputs.
            virtual Vector<T> calc(const  std::vector<std::reference_wrapper<Input<T>>>& input) const {                
                //dimension of the input vector
                size_t size=0;
                for (auto &i:input){
                    size+=i.get().size();
                }
                //the output vector has enough element buffer but zero length
                std::shared_ptr<T> data(new T[size]);
                Vector<T> r(data,0);
                for (auto &i:input) {
                    //append to the output vector
                    Vector<T>::append(i.get().get(), r);
                }
                return r;
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
                T* t=input.data().get();
                T max=t[0];
                for(size_t i=1;i<size;++i){
                    if(max<t[i]){
                        max=t[i];
                    }
                }
                double expSum=0;
                for(size_t i=0;i<size;++i){
                    expSum+=exp(t[i]-max);
                }
                double logExpSum=max + log(expSum);
                T* a=new T[size];
                for(size_t i=0;i<size;++i){
                    a[i]=exp(t[i] - logExpSum);
                }               
                //optimized by compiler so that no temporal object is generated
                return Vector<T>(std::shared_ptr<T>(a),size);
            }
    };
    
    //Defines a neural network compuation network.
    template <class T, class I>
    class NN {
        public:
            virtual Vector<T> calc(const I& input) const = 0;            
    };

    //Defines a multiple layer neural network, consisting of an input layer and a list of other layers.
    template <class T>
    class MLPNN: public NN<T, std::vector<std::reference_wrapper<Input<T>>>> {
        public:
            virtual Vector<T> calc(const std::vector<std::reference_wrapper<Input<T>>>& input) const {
                //thread safe
                Vector<T> v = m_inputLayer.calc(input);
                for(auto &layer: m_layers){
                    v=layer.get().calc(v);
                }
                return v;
            }

        public:
            MLPNN(const InputLayer<T>& inputLayer, const std::vector<Layer<T, Vector<T>>>& layers):m_inputLayer(inputLayer), m_layers(layers){}
            MLPNN(const MLPNN<T>& other):m_inputLayer(other.m_inputLayer), m_layers(other.m_layers){}
            MLPNN(MLPNN<T>&& other):m_inputLayer(std::move(other.m_inputLayer)), m_layers(std::move(other.m_layers)){}
                        
        private:
            const InputLayer<T> m_inputLayer;
            const std::vector<Layer<T, Vector<T>>> m_layers;
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
