#ifndef __NN__
#define __NN__

#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <memory>
#include <cmath>

<cstring>
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

template<typename T>
struct UnaryFunc {
    virtual T operator() (const T&) const;
};

template<typename T>
struct BinaryFunc {
    virtual T operator()(const T&, const T&) const;
};

template <class T>
class Vector {   
    public:
        void plus(const Vector<T>& v){
            ASSERT(v.m_size==m_size,"v");
            T* a = m_data.get();
            const T* b=v.m_data; 
            for(size_t i=0;i<m_size;++i){
                a[i]+=b[i];
            }
        }

        void apply(const UnaryFunc<T> func){
            ASSERT(func,"func");
            T* a = m_data.get();            
            for(size_t i=0;i<m_size;++i){
                a[i]+=func(a[i]);
            }   
        }  

 	T aggregate(const BinaryFunc<T> func, const T& t0) const {
            ASSERT(func,"func");
	    T t=t0;
            T* a = m_data.get();            
            for(size_t i=0;i<m_size;++i){
                t+=func(t,a[i]);
            }  
            return t;
        }
    
        size_t size() const {
            return m_size;
        } 
    
        const std::shared_ptr<T>& data() const {
            return m_data;
        } 

    public:
        static Vector<T> concatenate(const Vector<T>& v1, const Vector<T>& v2){
            size_t size = v1.m_size + v2.m_size;
            T* a = new T[size];
            std::shared_ptr<T> data(a);
            Vector<T> r(data, size);  
            append(v1,r);
            append(v2,r);
            return r;
        }

        static void append(const Vector<T>& v,  Vector<T>& r){
            if(v.m_size == 0){
                return;
            }                                 
            T* a = r.m_data.get() + r.m_size;
            memcpy (v.m_data.get(), a, sizeof(T)*v.m_size); 
            r.m_size += v.m_size;            
        }  

    public:
        Vector(const std::shared_ptr<T> data, const size_t size): m_data(data),m_size(size){}  
     
    private:
        const std::shared_ptr<T> m_data;
        const size_t m_size; 
};

template <class T>
class Matrix {
    public:
        Vector<T> multiply (const Vector<T>&  v) const {  
            ASSERT(m_col == v.size(),"v");
            T* a = v.data().get();
            T* y = new T[m_row];
            T* M = m_data.get();
            for(size_t i=0;i<m_row;++i){
                y[i] = 0;
                T* b = M[i*m_col];
                for(size_t j=0;j<m_col;++j){
                    y[i]+=a[j]*b[j];
                }
            } 
            std::shared_ptr<T> data(y);
            Vector<T> r(data, size);
            return r;               
        }   
    
        size_t row() const {
            return m_row;
        }

        size_t col() const {
            return m_col;
        }

        const std::shared_ptr<T>& data() const {
            return m_data;
        } 

    public:
        Matrix(const std::shared_ptr<T> data, const size_t row, const size_t col): m_data(data), m_row(row), m_col(col) {} 
     
    private:
	const std::shared_ptr<T> m_data;
	const size_t m_row;
	const size_t m_col;  
};

template<class T, class I>
public class Layer {
    public:
        virtual Vector<T> calc(const I& input) const = 0;  
};

template<class T>
class HiddenLayer: public Layer<T, Vector<T>> {
    public:
        //W: output*input matrix
        //b: bias vector with size being the number of output nodes 
        Layer(const Matrix<T>& W, const Vector<T>& b, const UnaryFunc<T>::type activationFunc): m_W(W), m_b(b),m_activationFunc(activationFunc){
            ASSERT(m_W.col() == b.size(),"W and b");
            ASSERT(m_activationFunc,"activationFunc");
        }

    public:
        virtual Vector<T> calc(const  Vector<T>& input) const {
            ASSERT(m_W.row() == input.size(),"input");
            Vector<T> r = m_W.multiply(input);
            r.plus(m_b);
            r.apply(m_activationFunc);
            return r;
        }
 
    private:
        Matrix<T> m_W;
        Vector<T> m_b;	
        const UnaryFunc<T>::type m_activationFunc;	
};

typedef unsigned int UINT;  

template<class T>
class Input {
    public:
        virtual Vector<T> get() const = 0;
        virtual size_t  size() const = 0;
    protected:
        Input(const Matrix<T>& embedding): m_embedding(embedding){}
    protected:        
        const Matrix<T> m_embedding;
};

const UINT PADDING_ID = 1;
const UINT UNK_ID = 0;

template<typename T>
struct Divide: public UnaryFunc<T> {
     Divide(const T denominator): m_denominator(denominator) {
         ASSERT(scalar != 0, "denominator");
     }   
     virtual T operator()(const T& t){
         return t/denominator;     
     }  
    private:
        const T m_denominator;   
};

template<class T>
class SequenceInput: public Input<T> {
    public:
        virtual Vector<T> get() const {
            //generate a vectors for each text window
            size_t size = m_embedding.col() * (2*m_contextLength + 1 );           
            std::shared_ptr<T> a(new T[size]);
            Vector<T> v(a,size);

            std::shared_ptr<T> b(new T[size]);
            Vector<T> r(b,size);
            memset (r.data().get(),0,sizeof(T)*size);

            for(UINT pos=0; pos < m_idSequence.size();++pos){
                generateConcatenatedVector(v,pos);
                r.plus(v);
            }   
            //average pooling
            Divide<T> divide(m_idSequence.size());
            r.apply(divide);
            return r;
        } 

        virtual size_t  size() const {
            return  m_embedding.col() * (2*m_contextLength + 1 );
        }   

    public:
        SequenceInput(const std::vector<UINT>& idSequence, const size_t contextLength, const Matrix<T> embedding): 
            Input<T>(embedding), m_idSequence(idSequence), m_contextLength(contextLength){
            ASSERT(m_idSequence.size() > 0,"idSequence");     
        }

    private:
        void generateConcatenatedVector(Vector<T>& v, UINT pos) const {
            T* E = m_embedding.data().get();
            T* c = v.data().get();
            size_t col = m_embedding.col(); 
            size_t size = m_idSequence.size();
            for(size_t i=pos-m_contextLength; i<=pos+m_contextLength;++i) {
                UINT id = i>=0 && i < size ? m_idSequence[i]:UINT PADDING_ID;
                memcpy(E+i*col, c, sizeof(T)*col);
                c+= col;
            }             
        }
          
    private:
	const std::vector<UINT> m_idSequence;
        const size_t m_contextLength;       
};

template<class T>
class NonSequenceInput: public Input<T> {
    public:
        virtual Vector<T> get() const {
            size_t size = m_embedding.col();
            T* a = new T[size]
            std::shared_ptr<T> data(a);            
            T* E = m_embedding.data().get();
            memcpy(E+m_id*col, a, sizeof(T)*size);
            Vector<T> r(data,size);
            return r;
        } 

        virtual size_t  size() const {
            return  m_embedding.col();
        } 

    public:
        NonSequenceInput(const UINT id, const Matrix<T> m_embedding):Input<T>(embedding),  m_id(id){
            ASSERT(id>=0 && id < embedding.row(),"id");
        }

    private:
	const UINT m_id;        
};


template<class T>
class InputLayer: public Layer<T, std::vector<Input<T>>> {
    public:
        virtual Vector<T> calc(const  std::vector<Input<T>>>& input) const {
            ASSERT(input.size() == m_embeddings.size(),"input"); 
            size_t size = 0;
            for ( auto &i : input ){
                size += i.size();  
            }            
            std::shared_ptr<T> data(new T[size]); 
            Vector<T> r(data,size);    
            for ( auto &i : input ) {
                Vector<T>.append(i.get(), r);
            }
            return r;
        }

    public:
        InputLayer(const std::vector<<Matrix<T>>>& embeddings): m_embeddings(embeddings){}

    private:
        std::vector<<Matrix<T>>> m_embeddings;     
};

template <class T>
class SoftmaxLayer: public Layer<T, Vector<T>> {
    public:
        virtual Vector<T> calc(const  Vector<T>& input) const {          
            ASSERT(input.size() > 0, "input"); 
            //ref to: http://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
            size_t size = input.size();
            T* t = input.data().get(); 
            T max = *t;;
            t++;
            for(size_t i=1;i<size;++i){
                if(max < *t){
                    max = *t;
                }
                t++;
            }
            t = input.data().get(); 
            double expSum = 0;
            for(size_t i=0;i<size;++i){
                expSum += exp(*t-max);
                t++; 
            } 
            double logExpSum = log(expSum);
            t = input.data().get();    
            T* a = new T[size]
            for(size_t i=0;i<size;++i){
                *a = exp(*t - logExpSum); 
                t++;
                a++;
            }  
            std::shared_ptr<T> data(a);            
            Vector<T> r(data,size);
            return r;
};


template <class T, class I>
class NN {
    public:
        virtual Vector<T> calc(const I& input) const = 0; 
};

template <class T>
class MLPNN: public NN<T, std::vector<Input<T>>> {
    public:
        virtual Vector<T> calc(const std::vector<Input<T>>& input) const {
             Vector<T> v = m_inputLayer.calc(input);
             for(auto &layer: m_layers){
                 v=layer.calc(v); 
             }
             return v;
        } 

    public:
        MLPNN(const InputLayer<T>& inputLayer, const std::vector<Layer<T, Vector<T>>>& layers):m_inputLayer(inputLayer), m_layers(layers){}
    private:
        InputLayer<T> m_inputLayer;
        std::vector<Layer<T, Vector<T>>> m_layers;
};

template<typename T>
struct Tanh: public UnaryFunc<T> {
     virtual T operator()(const T& t){
         return tanh(t);     
     }  
};

template<typename T>
struct ReLU: public UnaryFunc<T> {
     virtual T operator()(const T& t){
         return t>0?t:0;     
     }  
};

#endif
