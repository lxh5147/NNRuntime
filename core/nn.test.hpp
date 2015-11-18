/*
This file defines common functions.
*/
#ifndef __NN_TEST__
#define __NN_TEST__
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "nn.hpp"

namespace nn{
    namespace test{
        using namespace common;
        using namespace nn;
        using namespace std;
        using namespace boost;
        using namespace boost::random;

        template<typename T>
        shared_ptr<Vector<T> > newVectorWithDataArray(T data[], size_t size){
            T* pData=new T[size];
            memcpy(pData, data, size*sizeof(T) );
            return newVector(pData, size);
        }

        #define COMBINE1(X,Y) X##Y

        #define COMBINE(X,Y) COMBINE1(X,Y)

        #define DECL_VECTOR(T,v,size,...) T COMBINE(v##data,__LINE__)[]=__VA_ARGS__; Vector< T > v=*newVectorWithDataArray(COMBINE(v##data,__LINE__),size); 

        #define DECL_VECTOR_PTR(T,v,size,...) T COMBINE(v##data,__LINE__)[]=__VA_ARGS__; shared_ptr<Vector<T > > v=newVectorWithDataArray(COMBINE(v##data,__LINE__),size); 

        #define GET(v,i) ((v).data().get()[i])

        //assert two Vectors are equal
        #define ASSERT_EQUALS(T,v,total,...) {ASSERT((v).size()==total, #v);T COMBINE(_expected,__LINE__)[]=__VA_ARGS__;for(size_t i=0;i<total;++i){ASSERT(equals(GET((v),i),COMBINE(_expected,__LINE__)[i]),#v);}}

        template<typename T>
        shared_ptr<Matrix<T> > newMatrixWithDataArray(T data[], size_t row, size_t col){
            T* pData=new T[row*col];
            memcpy(pData, data, row*col*sizeof(T) );
            return newMatrix(pData, row,col);
        }

        #define DECL_MATRIX_PTR(T,m,row,col,...) T COMBINE(m##data,__LINE__)[]=__VA_ARGS__; shared_ptr<Matrix<T> > m=newMatrixWithDataArray(COMBINE(m##data,__LINE__),row,col); 

        #define DECL_MATRIX(T,m,row,col,...) T COMBINE(m##data,__LINE__)[]=__VA_ARGS__; Matrix<T> m=*newMatrixWithDataArray(COMBINE(m##data,__LINE__),row,col); 


        template<typename T>
        bool equals (Vector<T>& t1, Vector<T>& t2){
            if(t1.size()!=t2.size()){
                return false;
            }
            size_t size=t1.size();
            T* data1=t1.data().get();
            T* data2=t2.data().get();
            for(size_t i=0;i<size;++i){
                if(!nn::equals(data1[i],data2[i])){
                    return false;
                }
            }
            return true;
        }

        template<typename T>
        bool equals (Matrix<T>& m1, Matrix<T>& m2){
            if(m1.row()!=m2.row()){
                return false;
            }
            if(m1.col()!=m2.col()){
                return false;
            }
            size_t size=m1.row()*m1.col();
            T* data1=m1.data().get();
            T* data2=m2.data().get();
            for(size_t i=0;i<size;++i){
                if(!nn::equals(data1[i],data2[i])){
                    return false;
                }
            }
            return true;
        }

        template<typename T>
        void generateRandomNumbers(T* buffer, size_t size, T min=0, T max=1000){
            mt19937 rng;
            uniform_real_distribution<T> dist(min,max);
            for(size_t i=0;i<size;++i){
                buffer[i]=dist(rng);
            }
        }

        void generateRandomNumbers(vector<size_t>& buffer, size_t size, size_t min, size_t max){
            mt19937 rng;
            uniform_int_distribution<size_t> dist(min,max);
            for(size_t i=0;i<size;++i){
                buffer[i]=dist(rng);
            }
        }

        //Defines helper function to create shared pointer of vector without using external data buffer.
        template<typename T>
        shared_ptr<Vector<T> > newVector(size_t size){
            return make_shared_ptr(new Vector<T>( make_shared_ptr(new T[size]),size));
        }

        //Defines helper function to create shared pointer of matrix, without using external data buffer.
        template<typename T>
        shared_ptr<Matrix<T> > newMatrix(size_t row, size_t col){
            return make_shared_ptr(new Matrix<T>(make_shared_ptr(new T[row*col]),row,col));
        }

        #define DECL_STD_VECTOR(T,v,size,...) T COMBINE(v##data,__LINE__)[]=__VA_ARGS__; std::vector<T> v;for(size_t i=0;i<size;++i) v.push_back(COMBINE(v##data,__LINE__)[i]);
        
        #define INIT_STD_VECTOR(T,v,size,...) T COMBINE(v##data,__LINE__)[]=__VA_ARGS__;for(size_t i=0;i<size;++i) v.push_back(COMBINE(v##data,__LINE__)[i]);
    }
}

#endif
