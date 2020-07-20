#!/bin/bash

_PATH=$(dirname "$(realpath $0)")

# Load AWS credentials
source $_PATH/../aws-credentials.env

# Destroy the already created cluster using Flintrock
flintrock destroy production-line-performance
