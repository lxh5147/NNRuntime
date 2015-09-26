%module nn_runtime
%{
#include "nn_runtime.h"
%}

%include "std_vector.i"

namespace std {
   %template(DoubleVector) vector<double>;
   %template(IdsVector) vector< vector<size_t> >;
}

%include "nn_runtime.h"

