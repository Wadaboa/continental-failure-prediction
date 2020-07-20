#!/bin/bash

_PATH=$(dirname "$(realpath $0)")

# Compile and package app in a JAR file
sbt clean package

# Load AWS credentials
source $_PATH/../aws-credentials.env

# Copy the JAR file to the running cluster using Flintrock
flintrock copy-file production-line-performance \
	"target/scala-2.12/production-line-performance_2.12-1.0.jar" \
	"/home/ec2-user/"

# Copy S3 dependencies
flintrock run-command production-line-performance "[ -e '/home/ec2-user/spark/jars/aws-java-sdk-1.11.822.jar' ] && exit 0 || exit 1"
if [ $? -eq 1 ]; then
	flintrock copy-file production-line-performance \
	$_PATH/../bin/aws-java-sdk-1.11.822.jar \
	/home/ec2-user/spark/jars/
else
	echo "File aws-java-sdk-1.11.822.jar already present in the cluster, skipping copy."
fi

flintrock run-command production-line-performance "[ -e '/home/ec2-user/spark/jars/hadoop-aws-3.2.0.jar' ] && exit 0 || exit 1"
if [ $? -eq 1 ]; then
	flintrock copy-file production-line-performance \
	$_PATH/../bin/hadoop-aws-3.2.0.jar \
	/home/ec2-user/spark/jars/
else
	echo "File hadoop-aws-3.2.0.jar already present in the cluster, skipping copy."
fi

# Run the remote JAR using spark-submit
# $1 should be the output given by `flintrock describe`, field `master`
CMD="spark-submit --class PerformanceEvaluator /home/ec2-user/production-line-performance_2.12-1.0.jar --input-path s3a://production-line-performance/datasets/bosch-less.data --classifier-name DT"
echo "Executing command ${CMD}"
ssh -i $_PATH/../my-key-pair.pem ec2-user@$1 -t "${CMD}"

# flintrock describe | ggrep -oP '(?<=master: )[^ ]*'
#flintrock run-command --master-only production-line-performance "spark-submit --class PerformanceEvaluator /home/ec2-user/production-line-performance_2.12-1.0.jar --input-path s3a://production-line-performance/datasets/bosch-less.data --classifier-name DT"
