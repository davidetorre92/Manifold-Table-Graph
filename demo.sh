#!/bin/bash

CONFIG_PATH='./bin/config.py'
MAIN_PATH='./bin/main.py'

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

print_title() {
  local title="$1"
  local border=$(printf '*%.0s' {1..$((${#title}+4))})
  printf "\n%s\n* %s *\n%s\n\n" "$border" "$title" "$border"
}

print_title "Manifold Table Graph"
echo ''
echo '################################'
echo 'Options:'
echo '################################'

if [[ -f ${CONFIG_PATH} ]]; then
    cat ${CONFIG_PATH}
else
    echo "Error: no config file in ${CONFIG_PATH}"
    exit 1
fi

echo ''
echo '################################'

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

$PYTHON_CMD ${MAIN_PATH} -c ${CONFIG_PATH}
