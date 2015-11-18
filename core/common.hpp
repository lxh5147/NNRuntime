/*
This file defines common functions.
*/
#ifndef __COMMON__
#define __COMMON__

#include <iostream>
#include <map>
#include <string>
#include <ctime>
#include <pthread.h>
#include <boost/shared_ptr.hpp>

namespace common{
    using namespace std;
    using namespace boost;

    //Defines an macro that logs error message and stops the current program if some condition does not hold.
    #define ASSERT(condition, message) \
        do { \
            if (! (condition)) { \
                std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                          << " line " << __LINE__ << ": " << message << std::endl; \
                exit(EXIT_FAILURE); \
            } \
        } while (false)

    //Defines a helper function to create shared pointer by leveraging the type inference capability of templated static function.
    template<typename T>
    shared_ptr<T> make_shared_ptr(T* p){
        ASSERT(p,"p");
        return shared_ptr<T>(p);
    }

    //Defines helper function to compare two values.
    template<typename T, typename U>
    bool equals (const T& t1, const U& t2){
        return abs(t1-t2) <= 0.000001;
    }

    //Defines iterator interface.
    template<class T>
    class Iterator{
        public:
            //If next element is available, it will return true and load next element into buffer; otherwise return false.
            virtual bool next(T&)=0;
    };

    //Defines cache.
    template<typename T,typename K=string>
    class Cache{
        public:
            //Gets cached item with its key.
            shared_ptr<T> get(const K& key) {
                shared_ptr<T> item;
                int retCode=pthread_rwlock_rdlock(&m_lock);
                ASSERT(retCode==0,"acquire read lock");
                if(m_items.find(key)!=m_items.end()){
                    item=m_items[key];
                }
                retCode=pthread_rwlock_unlock(&m_lock);
                ASSERT(retCode==0,"release read lock");
                return item;
            }
            //Puts an item to this cache.
            void put(const K& key, const shared_ptr<T>& item){
                int retCode=pthread_rwlock_wrlock(&m_lock);
                ASSERT(retCode==0,"aquire exclusive write lock");
                m_items[key]=item;
                retCode=pthread_rwlock_unlock(&m_lock);
                ASSERT(retCode==0,"release exclusive write lock");
            }
        public:
            Cache(){
                int retCode=pthread_rwlock_init(&m_lock, NULL);
                ASSERT(retCode==0,"initialize lock");
            }
        private:
            map<K, shared_ptr<T> > m_items;
            pthread_rwlock_t m_lock; 
    };

    template<typename T>
    class Cache<T,size_t>{
        public:
            //Gets cached item with its key.
            shared_ptr<T> get(const size_t& key) {
                ASSERT(key>0 && key<=m_items.size(),"key");
                int retCode=pthread_rwlock_rdlock(&m_lock);
                ASSERT(retCode==0,"acquire read lock");
                shared_ptr<T> item=m_items[key-1];
                retCode=pthread_rwlock_unlock(&m_lock);
                ASSERT(retCode==0,"release read lock");
                return item;
            }
            //Puts an item to this cache.
            size_t put(const shared_ptr<T>& item){
                int retCode=pthread_rwlock_wrlock(&m_lock);
                ASSERT(retCode==0,"aquire exclusive write lock");
                m_items.push_back(item);
                size_t key=m_items.size();
                retCode=pthread_rwlock_unlock(&m_lock);
                ASSERT(retCode==0,"release exclusive write lock");
                return key;
            }
        public:
            Cache(){
                int retCode=pthread_rwlock_init(&m_lock, NULL);
                ASSERT(retCode==0,"initialize lock");
            }
        private:
            vector<shared_ptr<T> > m_items;
            pthread_rwlock_t m_lock; 
    };

    //PERF measurement helpers
    #define microseconds(x) (((double)x)/ CLOCKS_PER_SEC * 1000000)

    //range for loop
    #define BEGIN_STD_VECTOR_FOR(v,it,...) for(size_t i=0;i<(v).size();++i){const __VA_ARGS__& it= (v)[i];

    #define END_STD_VECTOR_FOR }

}

#endif
