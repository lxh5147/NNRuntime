/*
This file defines the quantization routine.
*/
#ifndef __QUANTIZATION__
#define __QUANTIZATION__

#include <memory>

namespace quantization{

    using namespace std;
    using namespace common;

    //Defines the quantization interface
    template<typename T, typename Q>
    class Quantizer {
        public:
            virtual Q quantize(T t) const=0;
            virtual T unquantize(Q q) const=0;
        public:
            virtual ~Quantizer(){}
    };

    template<typename T>
    void calcMinMax(Iterator<T>& it, T& min, T& max){
        T cur;
        auto hasElement=it.next(cur);
        ASSERT(hasElement,"hasElement");
        min=cur;
        max=cur;
        while(it.next(cur)){
            if(cur<min){
                min=cur;
                continue;
            }
            if(cur>max){
                max=cur;
                continue;
            }
        }
    }

    //Defines linear quantization strategy: which uses the average values in an interval to represent values in that interval.
    template<typename T, typename Q>
    class LinearQuantizer: public Quantizer<T,Q> {
        public:
            //computation complexity: O(n), where n is the number of values to be quantized
            static shared_ptr<Quantizer<T,Q>> create(Iterator<T>& it, Q size, T min, T max){
                ASSERT(size>0,"size");
                ASSERT(max>=min,"max>=min");
                //initialized to zero
                T* values=new T[size]();
                ASSERT(values,"values");
                T interval=(max-min)/size;
                //initialized to zero
                size_t* counts=new size_t[size]();
                ASSERT(counts,"counts");
                T cur;
                Q quantized;
                while(it.next(cur)){
                    quantized=quantize(cur,min,interval);
                    if(quantized>=size){
                        quantized=size-1;
                    }
                    values[quantized]+=cur;
                    counts[quantized]+=1;
                }
                T halfInterval=interval/2;
                T defaultIntervalValue=min+halfInterval;
                for(size_t i=0;i<size;++i){
                    if(counts[i]>0){
                        values[i]/=counts[i];
                    }else{
                        values[i]=defaultIntervalValue;
                    }
                    defaultIntervalValue+=interval;
                }
                delete[] counts;
                return make_shared_ptr(new LinearQuantizer(make_shared_ptr(values),size,min,interval));
            }
        private:
            static inline Q quantize(T t, T min, T interval){
                return (Q)((t-min)/interval);
            }
        public:
            Q quantize(T t) const {
                ASSERT(t>=m_min,"t");
                Q q=quantize(t,m_min,m_interval);
                ASSERT(q<=m_size,"q");
                //in case t=max, left open
                if(q==m_size){
                   q=m_size-1;
                }
                return q;
            }
            T unquantize(Q q) const {
                ASSERT(q>=0&&q<m_size,"q");
                return m_values.get()[q];
            }
        private:
            LinearQuantizer(shared_ptr<T> values,Q size, T min, T interval):m_values(values),m_size(size),m_min(min),m_interval(interval){}
        private:
            const shared_ptr<T> m_values;
            const Q m_size;
            const T m_min;
            const T m_interval;
    };

    template<class T> using _16BitsLinearQuantizer = LinearQuantizer<T,unsigned short>;
    template<class T> using _8BitsLinearQuantizer = LinearQuantizer<T,unsigned char>;
    
    //More advanced quantization based on Fisher's Natural Breaks Classificatio, with a computation complexity of O(k×n×log(n)) 
    //http://wiki.objectvision.nl/index.php/Fisher's_Natural_Breaks_Classification#Dynamic_programming_approach
    template <typename T>
    constexpr T MaxValue(){
        ASSERT(false,"un supported max value");
        return -1;
    }

    template<>
    constexpr unsigned char MaxValue(){
        return 255;
    }

    template<>
    constexpr unsigned short MaxValue(){
        return 65535;
    }
}

#endif
