#!/bin/bash

# Check for input variables
if [ ! -z "$1" ] && [ "$1" != "compile" ] && [ "$1" != "no-compile" ]; then
	echo "Usage: $0 [COMPILE] [DEPLOY_MODE] [REMOTE_TYPE]"
	echo "Please choose COMPILE in {compile, no-compile}"
	exit -1
fi

if [ ! -z "$2" ] && [ "$2" != "remote" ] && [ "$2" != "local" ]; then
	echo "Usage: $0 [COMPILE] [DEPLOY_MODE] [REMOTE_TYPE]"
	echo "Please choose DEPLOY_MODE in {local, remote}"
	exit -1
fi

if [ ! -z "$3" ] && [ "$3" != "ssh" ] && [ "$3" != "flintrock" ] && [ "$3" != "cluster" ]; then
	echo "Usage: $0 [COMPILE] [DEPLOY_MODE] [REMOTE_TYPE]"
	echo "Please choose REMOTE_TYPE in {ssh, flintrock, cluster}"
	exit -1
fi

# Set input variables
COMPILE=$1
DEPLOY_MODE=$2
REMOTE_TYPE=$3

# Set default values
if [ -z "$COMPILE" ]; then
	COMPILE="no-compile"
fi

if [ -z "$DEPLOY_MODE" ]; then
	DEPLOY_MODE="local"
fi

if [ $DEPLOY_MODE == "remote" ] && [ -z "$REMOTE_TYPE" ]; then
	REMOTE_TYPE="ssh"
fi

# Set useful variables
_PATH=$(dirname "$(realpath $0)")
APP_NAME="production-line-performance"
MAIN_JAR_NAME="${APP_NAME}_2.12-1.0.jar"
MAIN_JAR_LOCAL_PATH="$_PATH/../target/scala-2.12/$MAIN_JAR_NAME"

# Spark submit and scala package arguments
MAIN_CLASS="--class main.BoschEvaluator"
DATASET="datasets/bosch/bosch-less-less.data"
MODEL="models/bosch/bosch-less-less"
CLASSIFIER_NAME="--classifier-name RF"

# Compile and package app in a JAR file
if [[ $COMPILE == "compile" ]]; then
	echo "Compiling project..."
	sbt clean package
	echo ""
fi

# Parse deploy mode
if [[ $DEPLOY_MODE == "remote" ]]; then
	# Load AWS credentials
	source $_PATH/../aws-credentials.env

	# Set useful variables
	EC2_NAME=$(flintrock describe | grep -oP '(?<=master: )[^ ]*')
	EC2_CLUSTER=$(flintrock describe --master-hostname-only | grep -oP '(?<=INFO  - )[^ ]*' | sed 's/.$//')
	EC2_USER="ec2-user"
	EC2_HOME="/home/$EC2_USER"
	EC2_SPARK_HOME="$EC2_HOME/spark"
	S3_BUCKET_LINK="s3a://$EC2_CLUSTER"

	# Define submit remote parameters
	MASTER="--master spark://${EC2_NAME}:7077"
	INPUT_PATH="--input-path $S3_BUCKET_LINK/$DATASET"
	MODEL_FOLDER="--model-folder $S3_BUCKET_LINK/$MODEL"
	PARAMS="$MASTER $MAIN_CLASS $EC2_HOME/$MAIN_JAR_NAME $INPUT_PATH $MODEL_FOLDER $CLASSIFIER_NAME"

	# Copy the JAR file to the running cluster using Flintrock, if re-compiled
	if [[ $COMPILE == "compile" ]]; then
		echo "Copying $MAIN_JAR_NAME to cluster..."
		flintrock copy-file $EC2_CLUSTER $MAIN_JAR_LOCAL_PATH "$EC2_HOME/$MAIN_JAR_NAME"
		echo ""
	fi

	# Remote launch configurations
	if [[ $REMOTE_TYPE == "ssh" ]]; then
		# Submit in remote machine, using SSH
		ssh -i $_PATH/../my-key-pair.pem $EC2_USER@$EC2_NAME -t "spark-submit ${PARAMS}"
	elif [[ $REMOTE_TYPE == "flintrock" ]]; then
		# Submit in remote machine, using Flintrock
		flintrock run-command --master-only $EC2_CLUSTER "spark-submit ${PARAMS}"
	elif [[ $REMOTE_TYPE == "cluster" ]]; then
		# Submit from local machine
		spark-submit --deploy-mode cluster ${PARAMS}
	fi
elif [[ $DEPLOY_MODE == "local" ]]; then
	# Define submit local parameters
	INPUT_PATH="--input-path $DATASET"
	MODEL_FOLDER="--model-folder $MODEL"
	PARAMS="$MAIN_CLASS $MAIN_JAR_LOCAL_PATH $INPUT_PATH $MODEL_FOLDER $CLASSIFIER_NAME"

	# Launch Spark locally
	spark-submit ${PARAMS}
fi
