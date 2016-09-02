#!/bin/bash
BIN_PATH=./bin
BIN_TESTS=`ls ${BIN_PATH}`

if [ -z ${BIN_TESTS} ]; then
    echo "No tests in directory ${BIN_PATH}"
    exit -1
fi
echo "Following tests will be executed:"
echo "############################"
for BIN in `ls ${BIN_PATH}`; do
    echo $BIN
    CUDACHECK=`ldd ${BIN_PATH}/$BIN |grep cublas`
    if [ -n "${CUDACHECK}" ]; then
        MEMCHECK=cuda-memcheck
    else
        MEMCHECK=valgrind
    fi
done
echo "We use ${MEMCHECK} to check tests."
echo "############################"

rm ${MEMCHECK}.log -f 2> /dev/null
for BIN in `ls ${BIN_PATH}`; do
    echo "Now testing ${BIN}:" 2>&1 | tee -a ${MEMCHECK}.log
    if [ "$MEMCHECK" == "valgrind" ]; then
        valgrind --track-origins=yes --leak-check=full ./bin/$BIN 2>&1 |tee -a ${MEMCHECK}.log
    elif [ "$MEMCHECK" == "cuda-memcheck" ]; then
        cuda-memcheck ./bin/$BIN 2>&1 |tee -a ${MEMCHECK}.log
    fi
done
echo "############################"
echo "Tests are over, more details are in ${MEMCHECK}.log file."
echo "############################"
