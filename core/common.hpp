/*
This file defines common functions.
*/
#ifndef __COMMON__
#define __COMMON__

#include <iostream>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <chrono>


namespace common{
    using namespace std;

    //PERF measurement helpers
    typedef chrono::high_resolution_clock CLOCK;
    #define microseconds(x) (std::chrono::duration_cast<chrono::microseconds>(wctduration).count())

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
    template<typename T>
    class Cache{
        public:
            //Gets cached item with its key.
            shared_ptr<T> get(const string& key) {
                shared_ptr<T> item=nullptr;
                lock.lock();
                if(items.find(key)!=items.end()){
                    item=items[key];
                }
                lock.unlock();
                return item;
            }
            //Puts an item to this cache.
            void put(const string& key, const shared_ptr<T>& item){
                lock.lock();
                items[key]=item;
                lock.unlock();
            }
        private:
            map<string, shared_ptr<T>> items;
            mutex lock;
    };
}

#endif
