#!/bin/bash

_PATH=$(dirname "$(realpath $0)")

# Compile and package app in a JAR file
sbt clean package

# Parse deploy mode
if [[ "$1" == "remote" ]]; then
	# Define submit common commands
	PARAMS="--class main.BoschEvaluator /home/ec2-user/production-line-performance_2.12-1.0.jar --input-path s3a://production-line-performance/datasets/bosch/bosch-less-less.data --classifier-name RF"

	# Load AWS credentials
	source $_PATH/../aws-credentials.env

	# Copy the JAR file to the running cluster using Flintrock
	flintrock copy-file production-line-performance \
		"target/scala-2.12/production-line-performance_2.12-1.0.jar" \
		"/home/ec2-user/"

	# Copy S3 dependencies
	flintrock run-command production-line-performance "[ -e '/home/ec2-user/spark/jars/aws-java-sdk-1.7.4.jar' ] && exit 0 || exit 1"
	if [ $? -eq 1 ]; then
		flintrock copy-file production-line-performance \
			$_PATH/../bin/aws-java-sdk-1.7.4.jar \
			/home/ec2-user/spark/jars/
	else
		echo "File aws-java-sdk-1.7.4.jar already present in the cluster, skipping copy."
	fi

	flintrock run-command production-line-performance "[ -e '/home/ec2-user/spark/jars/hadoop-aws-2.7.2.jar' ] && exit 0 || exit 1"
	if [ $? -eq 1 ]; then
		flintrock copy-file production-line-performance \
			$_PATH/../bin/hadoop-aws-2.7.2.jar \
			/home/ec2-user/spark/jars/
	else
		echo "File hadoop-aws-2.7.2.jar already present in the cluster, skipping copy."
	fi

	# Get master DNS name
	EC2_NAME=$(flintrock describe | ggrep -oP '(?<=master: )[^ ]*')

	if [[ "$2" == "ssh" ]]; then
		# Submit in remote machine, using SSH
		ssh -i $_PATH/../my-key-pair.pem ec2-user@$EC2_NAME -t "spark-submit ${PARAMS}"
	elif [[ "$2" == "flintrock" ]]; then
		# Submit in remote machine, using Flintrock
		flintrock run-command --master-only $EC2_NAME "spark-submit ${PARAMS}"
	else
		# Submit from local machine
		spark-submit \
			--master spark://${EC2_NAME}:7077 \
			--deploy-mode cluster \
			${PARAMS}
	fi
else
	# Launch Spark locally
	spark-submit \
		--class main.BoschEvaluator \
		$_PATH/../target/scala-2.12/production-line-performance_2.12-1.0.jar \
		--input-path $_PATH/../datasets/bosch/bosch-less-less.data \
		--classifier-name RF
fi
