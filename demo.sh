#!/bin/bash

# Function to compare Python versions
compare_python_versions() {
    [[ $1 == $2 ]] && return 1
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++)); do
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 2
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 3
        fi
    done
    return 1
}

echo 'GraphCluster DEMO'
echo 'Options:'

if [[ -f config.ini ]]; then
    cat config.ini
else
    echo "Error: config.ini not found."
    exit 1
fi

echo

# Check for Python and decide whether to use python or python3
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    PYTHON3_VERSION=$(python3 --version | awk '{print $2}')
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
else
    echo "Error: Python is not installed."
    exit 1
fi

# If both python and python3 are available, compare versions
if [[ -n "$PYTHON3_VERSION" && -n "$PYTHON_VERSION" ]]; then
    compare_python_versions $PYTHON3_VERSION $PYTHON_VERSION
    case $? in
        2) PYTHON_CMD=python3 ;;
        3) PYTHON_CMD=python ;;
    esac
fi

echo "Using $PYTHON_CMD"
echo '1) Distances evaluation'
$PYTHON_CMD bin/graph_definition/evaluate_similarity.py -c config.ini
if [ $? -ne 0 ]; then
    echo "Error in 'Distances evaluation'."
    exit 1
fi

echo '2) Graph creation'
$PYTHON_CMD bin/graph_definition/graph_creation.py -c config.ini
if [ $? -ne 0 ]; then
    echo "Error in 'Graph creation'."
    exit 1
fi

echo '3) Graph EDA'
$PYTHON_CMD bin/tasks/visualization.py -c config.ini
if [ $? -ne 0 ]; then
    echo "Error in 'Graph EDA'."
    exit 1
fi
