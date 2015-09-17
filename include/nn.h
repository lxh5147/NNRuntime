#ifndef __NN__
#define __NN__

template <class T>
class Matrix {
    public:
        Matrix(T* data, size_t row, size_t col);
        ~Matrix(); 	
        T* Get(size_t i);  
         
    private:
	T* m_data;
	size_t m_row;
	size_t m_col;  
};

template <class T>
class Vector {
    public:
        Vector(T* data, size_t size); 
        ~Vector();        
    private:
        T* m_data;
        size_t m_size;    	
};




template<typename T>
struct UnaryFunc {
    typedef T (*type)(T);
};

template<typename T>
struct BinaryFunc {
    typedef T (*type)(T t1, T t2);
};

// v = f(v), element wise application of f
template<typename T> void Apply(UnaryFunc<T>::type func, size_t size, T* v);

//a = a + b
template<typename T> void Plus(size_t size, T* a, const T* b); 

//r = m*v, m: r*c matrix; v: c dimension vector; r: r dimension vector
template<typename T>   Multiply (const Matrix<T>& m, const T*  v, T* r); 

//aggregate all elements in a vector
template<typename T> T Aggregate(BinaryFunc<T>::type func, const T t0,  const Vector<T>& v); 

//r=concatentae(v1,v2)
template<typename T> void Concatenate(const Vector<T>& v1, const Vector<T>& v2, Vector<T>& r);

template<class T>
class Layer {
    public:
        //weight: output*input matrix
        //b: bias vector with size being the number of output nodes 
        Layer(const size_t input, const size_t output, T* weights, T* b, UnaryFunc<T>::type activationFunc); 
        ~Layer(); 
        void Calc (const T* input, T* output);  
    private:
        Vector<T> m_b;
	Matrix<T> m_W;
        UnaryFunc<T>::type m_activationFunc;	
};

template<class T>
class InputLayer {
    private:
        
}

#endif
