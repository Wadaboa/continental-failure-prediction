# Production Line Performance

This repository contains the implementation of a solution to the [Bosch Production Line Performance Competition](https://www.kaggle.com/c/bosch-production-line-performance), hosted on Kaggle in 2016. The solution is highly based on the analysis presented in [[1]](#1) and [[2]](#2) and has been implemented using the Scala/Spark combo.

## Nested projects

In order to test Spark's capabilities and to try ad-hoc functions, other datasets have been included in the project, even though they are not related at all to the main problem, i.e. the `Bosch Production Line Performance Competition`. The extra datasets are the following:
- [Adult](https://archive.ics.uci.edu/ml/datasets/adult): Aims at separating people whose income is greater than 50 thousands dollars per year from the rest. This dataset was useful to test various classifiers (e.g. decision trees, random forests) and check if the custom evaluation functions (e.g. [Matthew's Correlation Coeffiecient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)) were correct.
- [Arrest](https://www.kaggle.com/deepakg/usarrests): Contains statistics, in arrests per 100.000 residents, for assault, murder, and rape in each of the 50 US states in 1973. It also gives the percent of the population living in urban areas. This dataset was useful to test various clustering algorithms (mainly k-means) and check if the custom evaluation functions (`Inertia` and `Gap` [[5]](#5)) were correct.

## Environment info

Package versions in use:
- **SBT**: `1.3.13`
- **Scala**: `2.12.10`
- **Scala OpenJDK**: `1.8`
- **Spark**: `3.0.0`
- **Spark OpenJDK**: `11`
- **Flintrock**: `1.0.0`
- **Python** (required by Flintrock): `3.8.4`

## Installation & Execution

### Local

Compilation:
```bash
sbt clean package
```

Execution:
```bash
spark-submit \
	--class "main.<SomeEvaluator>" \
	"target/scala-2.12/production-line-performance_2.12-1.0.jar"\
	--input-path "<dataset-path>" \
	--classifier-name "<classifier-name>"
```

Here, `<SomeEvaluator>` is the main you want to run (check the `src/main/scala/main/` package for selection); `<dataset-path>` is the path to the dataset you want to use (check the `datasets/` folder for selection); and `<classifier-name>` is the name of the classifier you want to exploit (check the section below for selection).

### Remote

**This project makes use of the [Flintrock](https://github.com/nchammas/flintrock) package to manage EC2 clusters and install HDFS/Spark on them, instead of the simpler AWS EMR service, since EMR is not yet compatible with `Spark 3.0.0`, at the time of writing (July 2020).**

In order to run the Scala/Spark application remotely, on an AWS EC2 cluster, you need to do the following:
1. Make sure that you have generated an EC2 key pair (AWS Console → EC2 → Key pairs → Create a key pair → Name it `my-key-pair`) and saved the `.pem` file in the root of the project, as `my-key-pair.pem`.
2. Make sure that your AWS credentials are correctly set inside the `aws-credentials.env` file in the root of the project (see the `Notes` section for more info). Below you can see a template for the `aws-credentials.env` file:
```bash
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN
```
3. Make sure that you have created a IAM role, named `ec2-s3`, with service `ec2` and policies `AmazonS3FullAccess`, `AmazonElasticMapReduceforEC2Role`.
4. Run the script `script/create-aws-cluster.sh`, which will create a cluster named `production-line-performance`.
5. Make sure that you have created an S3 bucket, named `production-line-performance`, containing the `datasets/` folder present in the root of the project.
6. Run the script `script/run-app.sh remote` and check the execution at `<ec2-master>:8080`, where `<ec2-master>` is the DNS name of the EC2's cluster master node (you can check it using the command `flintrock describe`, field `master`).

_Note_: You can also use the `script/run-app.sh` script to run the application locally. To do so, you just have to launch `script/run-app.sh`, without passing any additional parameter.

## Supported classifiers

The following is a list of supported classifiers (the highlighted name is the one that should be used with the `--classifier-name` main option):
- [Decision Tree](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-trees): `DT`
- [Random Forest](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier): `RF`
- [Gradient Boosted Tree](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier): `GBT`

## Other files/folders

The following is a description of files/folder which were not mentioned above:
- `bin/`: It contains JARs which are needed to access S3 services on an EC2 cluster.
- `slides/`: It contains the project's presentation [slides](slides/slides.pdf).
- `.java-version`: It contains the Java version to be used by [jenv](https://github.com/jenv/jenv), which is a Java environment manager.
- `.scalafmt.conf`: It contains the version of the [Scalafmt](https://scalameta.org/scalafmt/) Scala code formatter.
- `.flintrock-config.yaml`: It contains Flintrock's configurations, used in the creation of a new EC2 cluster.

## Notes
* AWS Educate account credentials change after 3 hours (the given session expires). In order to get them again you have to log in to [AWS Educate](https://aws.amazon.com/it/education/awseducate/), click on `AWS Account` and on the orange button labeled as `AWS Educate Starter Account`, which will take you to [Vocareum](https://labs.vocareum.com). On `Vocareum`, you can either go the [AWS Console](https://console.aws.amazon.com/) or copy/paste the session keys (`Account Details` section) into `~/.aws/credentials` (to use [awscli](https://aws.amazon.com/it/cli/)), or in `aws-credentials.env` (for this project).
* AWS Educate account supports only the following EC2 instance types: `t2.small`, `t2.micro`, `t2.nano`, `m4.large`, `c4.large`, `c5.large`, `m5.large`, `t2.medium`, `m4.xlarge`, `t2.nano`, `c4.xlarge`, `c5.xlarge`, `t2.2xlarge`, `m5.2xlarge`, `t2.large`, `t2.xlarge`, `m5.xlarge`.
* AWS Educate account supports only services in `us-east-1` region.
* AWS Educate account highly limits IAM management.
* Run `sbt console` to package the application and try it in the Scala REPL.

## References

- <a id="1">[1]</a>
  _Darui Zhang, Bin Xu, Jasmine Wood (2016)_.\
  **Predict Failures in Production Lines. A Two-stage Approach with Clustering and Supervised Learning**.\
  2016 IEEE International Conference on Big Data.
- <a id="2">[2]</a>
  _Caoimhe M Carbery, Roger Woods and Adele H Marshall (2019)_.\
  **A new data analytics framework emphasising preprocessing of data to generate insights into complex manufacturing systems**.\
  Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science, Volume 233 Issue 19-20, October 2019.
- <a id="3">[3]</a>
  _Ankita Mangal, Nishant Kumar (2016)_.\
  **Using Big Data to Enhance the Bosch Production Line Performance: A Kaggle Challenge**.\
  2016 IEEE International Conference on Big Data.
- <a id="4">[4]</a>
  _Aayush Mudgal, Sheallika Singh, Vibhuti Mahajan (2014)_.\
  **Reducing Manufacturing Failures**.\
  Columbia University E6893 Big Data Analytics Fall 2014 Final Report.
- <a id="5">[5]</a>
  _Robert Tibshirani, Guenther Walther, Trevor Hastie (2002)_.\
  **Estimating the number of clusters in a data set via the gap statistic**.\
  Journal of the Royal Statistical Society, Statistical Methodology, Series B, Volume 63 Issue 2, Pages 411-423.
