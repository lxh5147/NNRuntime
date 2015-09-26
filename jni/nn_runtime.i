%module nn_runtime

%{
#include "nn_runtime.h"
%}

%include "std_vector.i"
namespace std {
   %template(IdVector) vector<size_t>;
   %template(IdVectors) vector<vector<size_t>>;
}

%include "nn_runtime.h"
