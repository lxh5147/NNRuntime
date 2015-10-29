#!/bin/bash

#test code coverage profile
echo "run code coverage profiling"
prog_to_profile=test_with_profile
g++ -std=c++11 -g -O0 --coverage -o ${prog_to_profile}_coverage test.cpp 
lcov --zerocounters  --directory .
./${prog_to_profile}_coverage
lcov --directory . --capture --output-file ${prog_to_profile}_coverage.info
genhtml --output-directory coverage \
  --demangle-cpp --num-spaces 2 --sort \
  --title "$prog_to_profile Test Coverage" \
  --function-coverage --branch-coverage --legend \
  ${prog_to_profile}_coverage.info

#cpu profile with gprof
echo "run gprof cpu profiling"
g++ -std=c++11 -g -pg -o ${prog_to_profile}_gprof test.cpp
./${prog_to_profile}_gprof
gprof -b ${prog_to_profile}_gprof gmon.out > cpu_analysis.txt 

#memory profile using valgrind
echo "run massif memory profiling"
g++ -std=c++11 -g -o ${prog_to_profile}_massif test.cpp 
rm massif.out.*
valgrind --tool=massif  ./${prog_to_profile}_massif
ms_print massif.out.* > memory_analysis.txt 

