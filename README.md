# NNRuntime
A simple c++ neural network runtime that supports NN bag of words model, fully connected hidden layers and softmax output layer. It is designed for production usage, highly efficent in memory usage.

Q&A

1. How to build and test the project?
    ssh nrg5-unn4.nrg5.us.grid.nuance.com
    alias cmake='/nrg5/nlu/data/users/xiaoliu/cmake/cmake-3.4.0/bin/cmake -DCMAKE_C_COMPILER=/nrg5/nlu/data/users/xiaoliu/gcc/gcc-4.9/bin/gcc -DCMAKE_CXX_COMPILER=/nrg5/nlu/data/users/xiaoliu/gcc/gcc-4.9/bin/g++'
    export JAVA_HOME=/nlu/tools/opt/jdk1.7.0_21
    git clone git@git.labs.nuance.com:xiaohua.liu/NNRuntime.git
    cd NNRuntime
    cmake .
    make
    make test
    #perf test with the zoe_random_nbow-81.model.bin
    cd core
    ./nn_runtime_core_perf_test perfReal

2. How to run gpu based matrix vector multiplier
    #log on one host with GPU installed, and repeat the same process
    telnet gna-r01r05u01b02.nrg5.us.grid.nuance.com

3. How to run line coverage analysis and memory, cpu profiling for the core?
    cd core
    cmake .
    make
    ./profile.sh
    If you already build the project, it is safe to ignore "cmake ." and "make".
