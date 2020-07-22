package main

import preprocessing.{Dataset}
import prediction.{Predictor}

import org.apache.spark.sql.{SparkSession, DataFrame}
import preprocessing.Preprocessor

object AdultEvaluator {

  def main(args: Array[String]): Unit = {
    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")
    val classifierName = options.get("classifierName")

    // Create Dataset object (and read data)
    val dataset = Dataset(name = "Adult", inputPath = inputPath)
    dataset.show()

    // Preprocess data
    val preprocessed = dataset.preprocess()
    preprocessed.show()

    // Train the classifier and test it
    val classifier = Predictor(classifierName getOrElse "DT", dataset)
    classifier.train()
    val result = classifier.test()
    result.show()

    // Stop SparkSession execution
    stopSpark()
  }

}
