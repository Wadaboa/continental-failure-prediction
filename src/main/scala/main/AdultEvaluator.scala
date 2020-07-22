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
    println(dataset.maxDistinctValues)
    dataset.show()

    // Preprocess data
    val preprocessed = dataset.preprocess()
    preprocessed.show()

    // Train the classifier and test it
    val classifier = Predictor(classifierName getOrElse "DT", preprocessed)
    classifier.train(validate = true)
    val predictions = classifier.test()
    predictions.show()
    val result = classifier.evaluate(predictions)
    println(result)

    // Stop SparkSession execution
    stopSpark()
  }

}
