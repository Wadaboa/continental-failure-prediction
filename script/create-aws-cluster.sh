#!/bin/bash

_PATH=$(dirname "$(realpath $0)")

# Load AWS credentials
source $_PATH/../aws-credentials.env

# Edit PEM file permissions
chmod 400 $_PATH/../my-key-pair.pem

# Launch the cluster using Flintrock
flintrock --config $_PATH/../flintrock-config.yaml launch production-line-performance

# The following is a work-around to run spark-submit with the `--deploy-mode cluster` option,
# which assumes that you have installed OpenJDK 11 in your local machine at `/usr/local/opt/openjdk@11`
: '
flintrock run-command production-line-performance "sudo amazon-linux-extras install -y java-openjdk11"
flintrock run-command production-line-performance "yes 2 | sudo alternatives --config java"
flintrock run-command production-line-performance "sudo mkdir -p /usr/local/opt/openjdk@11/bin/"
flintrock run-command production-line-performance "sudo ln -s /usr/lib/jvm/jre/bin/java /usr/local/opt/openjdk@11/bin/"
'
