#!/bin/bash

# Check input variables
if [ -z "$1" ]; then
	CLUSTER_NAME=$(flintrock describe --master-hostname-only | grep -oP '(?<=INFO  - )[^ ]*' | sed 's/.$//')
else
	CLUSTER_NAME=$1
fi

# Define current path
_PATH=$(dirname "$(realpath $0)")

# Load AWS credentials
source $_PATH/../aws-credentials.env

# Destroy the already created cluster using Flintrock
flintrock destroy $CLUSTER_NAME
