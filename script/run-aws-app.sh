#!/bin/bash

_PATH=$(dirname "$(realpath $0)")

# Compile and package app in a JAR file
sbt clean package

# Copy the JAR file to the running cluster using Flintrock
flintrock copy-file production-line-performance \
	"target/scala-2.12/production-line-performance_2.12-1.0.jar" \
	"/home/ec2-user/"

# Run the remote JAR using spark-submit
# $1 should be the output given by `flintrock describe`, field `master`
spark-submit \
	--class "PerformanceEvaluator" \
	--master spark://$1:7077 \
	--deploy-mode cluster \
	"/home/ec2-user/production-line-performance_2.12-1.0.jar" \
	--input-path "s3a://production-line-performance/datasets/bosch-less.data" \
	--classifier-name "DT"
