#!/bin/bash

_PATH=$(dirname "$(realpath $0)")

# Load AWS credentials
source $_PATH/../aws-credentials.env

# Edit PEM file permissions
chmod 400 $_PATH/../my-key-pair.pem

# Launch the cluster using Flintrock
flintrock --config $_PATH/../flintrock-config.yaml launch production-line-performance
