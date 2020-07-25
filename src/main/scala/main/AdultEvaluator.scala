package main

import preprocessing.AdultDataset
import prediction.Predictor
import utils._

object AdultEvaluator {

  def main(args: Array[String]): Unit = {
    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")
    val classifierName = options.get("classifierName")

    // Create Dataset object (and read data)
    val dataset = AdultDataset(inputPath = inputPath)
    dataset.show()

    // Preprocess data
    val preprocessed = dataset.preprocess()
    preprocessed.show()

    // Train the classifier and test it
    val classifier = Predictor(classifierName getOrElse "DT", preprocessed)
    classifier.train(validate = false)
    val predictions = classifier.test()
    predictions.show()
    val accuracy = classifier.evaluate(predictions, metricName = "accuracy")
    val mcc = classifier.evaluate(predictions, metricName = "mcc")
    Logger.info(s"Accuracy score: ${accuracy}")
    Logger.info(s"MCC score: ${mcc}")

    // Stop SparkSession execution
    Spark.stop()
  }

}
