name := "production-line-performance"

version := "1.0"

scalaVersion := "2.12.10"

val sparkVersion = "3.0.0"
val hadoopAwsVersion = "3.1.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.hadoop" % "hadoop-aws" % hadoopAwsVersion
)
