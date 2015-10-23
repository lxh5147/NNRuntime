#include <vector>
using namespace std;

//Loads the binary model into memory. Returns the handle of the loaded model or HANDLE_INVALID if it failed to load the model.
extern size_t load(const char* modelPath,bool normalizeOutputWithSoftmax=true);

//Predicts a probability for each category. The input is a list of id sequences. For a non-sequence input, there must be one element in the id sequence.
//Note: leave a blank between vector< vector<> > to make swig happy
extern vector<double> predict(size_t modelHandle, const vector< vector<size_t> >& idsInputs);//

