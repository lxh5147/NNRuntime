#!/bin/bash

prog_to_profile=test_with_profile
g++ -std=c++11 -g -O0 --coverage -o $prog_to_profile test.cpp 
lcov --zerocounters  --directory .
./$prog_to_profile
lcov --directory . --capture --output-file $prog_to_profile.info
genhtml --output-directory coverage \
  --demangle-cpp --num-spaces 2 --sort \
  --title "$prog_to_profile Test Coverage" \
  --function-coverage --branch-coverage --legend \
  $prog_to_profile.info
