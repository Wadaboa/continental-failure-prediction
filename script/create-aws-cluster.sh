#!/bin/bash

# Check input variables
if [ -z "$1" ]; then
	CLUSTER_NAME="production-line-performance"
else
	CLUSTER_NAME=$1
fi

# Define current path
_PATH=$(dirname "$(realpath $0)")

# Load AWS credentials
source $_PATH/../aws-credentials.env

# Edit PEM file permissions
chmod 400 $_PATH/../my-key-pair.pem

# Launch the cluster using Flintrock
echo "Launching the cluster..."
flintrock --config $_PATH/../flintrock-config.yaml launch $CLUSTER_NAME
echo ""

# Set useful variables
EC2_USER="ec2-user"
EC2_HOME="/home/$EC2_USER"
EC2_SPARK_HOME="$EC2_HOME/spark"

# Copy S3 dependencies
AWS_SDK_JAR_PATH=$(find $_PATH/../bin -type f -maxdepth 1 -iname 'aws*')
AWS_SDK_JAR_NAME=$(basename -- "$AWS_SDK_JAR_PATH")
echo "Copying ${AWS_SDK_JAR_NAME} to cluster..."
flintrock copy-file $CLUSTER_NAME $AWS_SDK_JAR_PATH "$EC2_SPARK_HOME/jars/"
echo ""

HADOOP_AWS_JAR_PATH=$(find $_PATH/../bin -type f -maxdepth 1 -iname 'hadoop*')
HADOOP_AWS_JAR_NAME=$(basename -- "$HADOOP_AWS_JAR_PATH")
echo "Copying ${HADOOP_AWS_JAR_NAME} to cluster..."
flintrock copy-file $CLUSTER_NAME $HADOOP_AWS_JAR_PATH "$EC2_SPARK_HOME/jars/"
echo ""

# The following is a work-around to run spark-submit with the `--deploy-mode cluster` option,
# which assumes that you have installed OpenJDK 11 in your local machine at `/usr/local/opt/openjdk@11`
: '
flintrock run-command production-line-performance "sudo amazon-linux-extras install -y java-openjdk11"
flintrock run-command production-line-performance "yes 2 | sudo alternatives --config java"
flintrock run-command production-line-performance "sudo mkdir -p /usr/local/opt/openjdk@11/bin/"
flintrock run-command production-line-performance "sudo ln -s /usr/lib/jvm/jre/bin/java /usr/local/opt/openjdk@11/bin/"
'
