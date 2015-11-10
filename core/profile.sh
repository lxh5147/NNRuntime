#!/bin/bash


echo "run code coverage analysis"
prog_to_profile=nn_runtime_core_test
./${prog_to_profile}_coverage
#code coverage with lcov
: '
lcov --zerocounters  --directory .
lcov --directory . --capture --output-file ${prog_to_profile}_coverage.info
genhtml --output-directory coverage \
  --demangle-cpp --num-spaces 2 --sort \
  --title "$prog_to_profile Test Coverage" \
  --function-coverage --branch-coverage --legend \
  ${prog_to_profile}_coverage.info
'
#code coverage analysis with gcov, check nn.hpp.gcov
gcov test.cpp

#cpu profile with gprof
echo "run gprof cpu profiling"
perf_test_option=perfReal
./${prog_to_profile}_perf $perf_test_option
gprof -b ${prog_to_profile}_perf gmon.out > cpu_analysis.txt 

#memory profile using valgrind, please enable the following script after install valgrind and ms_print
: '
echo "run massif memory profiling"
perf_test_option=perf
rm massif.out.*
valgrind --tool=massif  ./${prog_to_profile} $perf_test_option
ms_print massif.out.* > memory_analysis.txt
'
