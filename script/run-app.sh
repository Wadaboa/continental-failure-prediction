#!/bin/bash

# Check for input variables
if [ ! -z "$1" ] && [ "$1" != "remote" ] && [ "$1" != "local" ]; then
	echo "Usage: $0 [DEPLOY_MODE] [REMOTE_TYPE]"
	echo "Please choose DEPLOY_MODE in {local, remote}"
fi

if [ ! -z "$2" ] && [ "$2" != "ssh" ] && [ "$2" != "flintrock" ] && [ "$2" != "cluster" ]; then
	echo "Usage: $0 [DEPLOY_MODE] [REMOTE_TYPE]"
	echo "Please choose REMOTE_TYPE in {ssh, flintrock, cluster}"
fi

# Set input variables
DEPLOY_MODE=$1
REMOTE_TYPE=$2

# Set default values
if [ -z "$DEPLOY_MODE" ]; then
	DEPLOY_MODE="local"
fi

if [ $DEPLOY_MODE == "remote" ] && [ -z "$REMOTE_TYPE" ]; then
	REMOTE_TYPE="flintrock"
fi

# Set useful variables
_PATH=$(dirname "$(realpath $0)")
EC2_CLUSTER=$(flintrock describe --master-hostname-only | grep -oP '(?<=INFO  - )[^ ]*' | sed 's/.$//')
EC2_USER="ec2-user"
EC2_HOME="/home/$EC2_USER"
EC2_SPARK_HOME="$EC2_HOME/spark"
S3_BUCKET_LINK="s3a://$EC2_CLUSTER"
MAIN_JAR_NAME="${EC2_CLUSTER}_2.12-1.0.jar"
MAIN_JAR_LOCAL_PATH="$_PATH/../target/scala-2.12/$MAIN_JAR_NAME"

# Spark submit and scala package arguments
MAIN_CLASS="--class main.BoschEvaluator"
DATASET="datasets/bosch/bosch-less-less.data"
MODEL="models/bosch/"
CLASSIFIER_NAME="--classifier-name RF"

# Compile and package app in a JAR file
sbt clean package

# Parse deploy mode
if [[ $DEPLOY_MODE == "remote" ]]; then
	# Define submit remote parameters
	INPUT_PATH="--input-path $S3_BUCKET_LINK/$DATASET"
	MODEL_FOLDER="--model-folder $S3_BUCKET_LINK/$MODEL"
	PARAMS="$MAIN_CLASS $EC2_HOME/$MAIN_JAR_NAME $INPUT_PATH $MODEL_FOLDER $CLASSIFIER_NAME"

	# Load AWS credentials
	source $_PATH/../aws-credentials.env

	# Copy the JAR file to the running cluster using Flintrock
	flintrock copy-file $EC2_CLUSTER $MAIN_JAR_LOCAL_PATH $EC2_HOME

	# Copy S3 dependencies
	AWS_SDK_JAR_PATH=$(find $_PATH/../bin -type f -maxdepth 1 -iname 'aws*')
	AWS_SDK_JAR_NAME=$(basename -- "$AWS_SDK_JAR_PATH")
	flintrock run-command $EC2_CLUSTER "[ -e '$EC2_SPARK_HOME/jars/$AWS_SDK_JAR_NAME' ] && exit 0 || exit 1"
	if [ $? -eq 1 ]; then
		flintrock copy-file $EC2_CLUSTER $AWS_SDK_JAR_PATH "$EC2_SPARK_HOME/jars/"
	else
		echo "File $AWS_SDK_JAR_NAME already present in the cluster, skipping copy."
	fi

	HADOOP_AWS_JAR_PATH=$(find $_PATH/../bin -type f -maxdepth 1 -iname 'hadoop*')
	HADOOP_AWS_JAR_NAME=$(basename -- "$HADOOP_AWS_JAR_PATH")
	flintrock run-command $EC2_CLUSTER "[ -e '$EC2_SPARK_HOME/jars/$HADOOP_AWS_JAR_NAME' ] && exit 0 || exit 1"
	if [ $? -eq 1 ]; then
		flintrock copy-file $EC2_CLUSTER $HADOOP_AWS_JAR_PATH "$EC2_SPARK_HOME/jars/"
	else
		echo "File $HADOOP_AWS_JAR_NAME already present in the cluster, skipping copy."
	fi

	# Get master DNS name
	EC2_NAME=$(flintrock describe | grep -oP '(?<=master: )[^ ]*')

	# Remote launch configurations
	if [[ $REMOTE_TYPE == "ssh" ]]; then
		# Submit in remote machine, using SSH
		ssh -i $_PATH/../my-key-pair.pem $EC2_USER@$EC2_NAME -t "spark-submit ${PARAMS}"
	elif [[ $REMOTE_TYPE == "flintrock" ]]; then
		# Submit in remote machine, using Flintrock
		flintrock run-command --master-only $EC2_CLUSTER "spark-submit ${PARAMS}"
	elif [[ $REMOTE_TYPE == "cluster" ]]; then
		# Submit from local machine
		spark-submit --master spark://${EC2_NAME}:7077 --deploy-mode cluster ${PARAMS}
	fi
elif [[ $DEPLOY_MODE == "local" ]]; then
	# Define submit local parameters
	INPUT_PATH="--input-path $DATASET"
	MODEL_FOLDER="--model-folder $MODEL"
	PARAMS="$MAIN_CLASS $MAIN_JAR_LOCAL_PATH $INPUT_PATH $MODEL_FOLDER $CLASSIFIER_NAME"

	# Launch Spark locally
	spark-submit ${PARAMS}
fi
