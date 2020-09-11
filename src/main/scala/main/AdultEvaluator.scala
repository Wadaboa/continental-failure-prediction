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
    val modelFolder = options.get("modelFolder")

    // Create Dataset object (and read data)
    val dataset = AdultDataset(inputPath = inputPath)
    dataset.show()

    // Preprocess data
    val preprocessed = dataset.preprocess()
    preprocessed.show()

    // Train the classifier and test it
    val classifier = Predictor(classifierName getOrElse "DT", preprocessed)
    val modelFile =
      s"${modelFolder.orNull.toString}/adult-${classifierName.orNull.toLowerCase}.ml"
    if (fileExists(modelFile)) classifier.load(modelFile)
    else classifier.train(validate = true)
    val predictions = classifier.test()
    predictions.show()

    // Compute and print evaluations
    var accuracy = classifier.evaluate(predictions, metric = "accuracy")
    var precision = classifier.evaluate(predictions, metric = "precision")
    var recall = classifier.evaluate(predictions, metric = "recall")
    var fscore = classifier.evaluate(predictions, metric = "f1")
    var mcc = classifier.evaluate(predictions, metric = "mcc")
    var auroc = classifier.evaluate(predictions, metric = "areaUnderROC")
    Logger.info(s"Accuracy score: ${accuracy}")
    Logger.info(s"Precision score: ${precision}")
    Logger.info(s"Recall score: ${recall}")
    Logger.info(s"F1 score: ${fscore}")
    Logger.info(s"Area under ROC score: ${auroc}")
    Logger.info(s"MCC score: ${mcc}")

    // Save the model
    if (!fileExists(modelFile)) classifier.save(modelFile)

    // Stop SparkSession execution
    Spark.stop()
  }

}
