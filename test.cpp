#include "nn.h"

void SoftmaxLayerTest(){
     float* t = new float[2];
     t[0]=0.5;
     t[1]=0.5;
     std::shared_ptr<float> data(t);     
     Vector<float> input(data,2);  
     //return input; 
     SoftmaxLayer<float> softmaxLayer;
     Vector<float> result = softmaxLayer.calc(input); 
     ASSERT(result.size() == 2, "result");
     std::cout << result.data().get()[0] << std::endl;
     ASSERT(result.data().get()[0] == 0.5, "result");
     ASSERT(result.data().get()[1] == 0.5, "result");
}

int main( int argc, const char* argv[] )
{
    SoftmaxLayerTest();
}
