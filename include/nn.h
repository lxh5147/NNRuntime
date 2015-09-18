#ifndef __NN__
#define __NN__

template<typename T>
struct UnaryFunc {
    typedef void (*type)(const T&, T&);
};

template<typename T>
struct BinaryFunc {
    typedef void (*type)(const T&, const T&, T&);
};

template <class T>
class Matrix {
    public:
        void Multiply (const Vector<T>&  v, Vector<T>& r) contst;
    public:
        Matrix(const std::shared_ptr<T> data, const size_t row, const size_t col);      
    private:
	const std::shared_ptr<T> m_data;
	const size_t m_row;
	const size_t m_col;  
};

template <class T>
class Vector {
    public:
        void Plus(const Vector<T>& b);
        void Apply(const UnaryFunc<T>::type func);
 	T Aggregate(const BinaryFunc<T>::type func, const T& t0) const;
    public:
        static Vector<T> Concatenate(const Vector<T>& v1, const Vector<T>& v2);
    public:
        Vector(const std::shared_ptr<T> data, const size_t size);       
    private:
        const std::shared_ptr<T> m_data;
        const size_t m_size;    	
};

template<class T, class I>
public class Layer {
    public:
        virtual Vector<T> Calc(const I& input) const = 0;  
};

template<class T>
class HiddenLayer: public Layer<T, Vector<T>> {
    public:
        //weight: output*input matrix
        //b: bias vector with size being the number of output nodes 
        Layer(const Matrix<T>& weights, const Vector<T>& b, const UnaryFunc<T>::type activationFunc); 
    private:
        Matrix<T> m_W;
        Vector<T> m_b;	
        const UnaryFunc<T>::type m_activationFunc;	
};

typedef unsigned int UINT;  

template<class T>
class Input {
    public:
        virtual Vector<T> Get() const =0;
    protected:
        Input(const Matrix<T>& embedding);
    protected:        
        const Matrix<T> m_embedding;
};

template<class T>
class SequenceInput: public Input<T> {
    public:
        SequenceInput(const std::vector<UINT>& idSequence, const size_t context, const Matrix<T> m_embedding);
    private:
	const std::vector<UINT> m_idSequence;
        const size_t m_contextLength;       
};

template<class T>
class NonSequenceInput: public Input<T> {
    public:
        NonSequenceInput(const UINT id, const Matrix<T> m_embedding);
    private:
	const UINT m_id;        
};


template<class T>
class InputLayer: public Layer<T, std::vector<Input<T>>> {
    public:
        InputLayer(const std::vector<<Matrix<T>>>& embeddings);
    private:
        std::vector<<Matrix<T>>> m_embeddings;     
};

template <class T>
class SoftmaxLayer: public Layer<T, Vector<T>> {
};


template <class T, class I>
class NN {
    public:
        virtual Vector<T> Calc(const I& input) const = 0; 
};

template <class T>
class MLPNN: public NN<T, std::vector<Input<T>>> {
    public:
        MLPNN(const InputLayer<T>& inputLayer, const std::vector<Layer<T, Vector<T>>>& layers);
    private:
        InputLayer<T> m_inputLayer;
        std::vector<Layer<T, Vector<T>>> m_layers;
};


#endif
