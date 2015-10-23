#include <vector>
using namespace std;

//Loads the binary model into memory. Returns the handle of the loaded model or HANDLE_INVALID if it failed to load the model.
//Parameters:modelPath represents the binary model stored in disk; normalizeOutputWithSoftmax indicates whether or not the outputs should be normalized with softmax
extern size_t load(const char* modelPath,bool normalizeOutputWithSoftmax=true);

//Predicts a probability for each category. The input is a list of id sequences. For a non-sequence input, there must be one element in the id sequence.
//Parameters: modelHandler represents the handle of the runtime model that has been loaded; idsInputs represents the list of id sequences which are the inputs of the runtime.
//Note: leave a blank between vector< vector<> > to make swig happy
extern vector<double> predict(size_t modelHandle, const vector< vector<size_t> >& idsInputs);

