package main

import preprocessing.{BoschDataset, Preprocessor}
import prediction.{Predictor, Clusterer}
import utils._

object BoschEvaluator {

  def main(args: Array[String]): Unit = {
    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")
    val classifierName = options.get("classifierName")

    // Create Dataset object (and read data)
    val dataset = BoschDataset(inputPath = inputPath)
    dataset.show()

    // Preprocess data
    val toCluster = dataset.preprocessForClustering()
    toCluster.show()

    // Cluster data and print statistics
    val kmeans = Clusterer("KM", toCluster)
    kmeans.maxClusters = 3
    kmeans.train()
    val predictions = kmeans.trainedModel.transform(toCluster.data)
    predictions.show()
    val clusterCenters = kmeans.trainedModel.clusterCenters
    val numClusters = clusterCenters.length
    val inertia = kmeans.evaluate(predictions, metricName = "inertia")
    val silhouette = kmeans.evaluate(predictions, metricName = "silhouette")
    Logger.info(s"Cluster centers: ${clusterCenters.mkString(" ")}")
    Logger.info(s"Inertia score: ${inertia(0)}")
    Logger.info(s"Silhouette score: ${silhouette(0)}")

    /*
    // Train the classifier and test it
    val classifier = Classifier(classifierName.orNull.toString, dataset)
    classifier.train()
    val result = classifier.test()
    println(result)
     */

    // Stop SparkSession execution
    Spark.stop()
  }

}
