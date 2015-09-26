#include <vector>

using namespace std;

extern int load(const char* modelPath);
extern vector<size_t> predict(int modelHandle, const vector<vector<size_t>>& idsInputs);
