package main

import preprocessing.ArrestDataset
import prediction.Clusterer
import utils._

object ArrestEvaluator {

  def main(args: Array[String]): Unit = {
    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")
    val modelFolder = options.get("modelFolder")

    // Create Dataset object (and read data)
    val dataset = ArrestDataset(inputPath = inputPath)
    dataset.show()

    // Preprocess data
    val toCluster = dataset.preprocess()
    toCluster.show()

    // Cluster data and print statistics
    val kmeans = Clusterer("KM", toCluster)
    val modelFile = s"${modelFolder.orNull.toString}/arrest-kmeans.ml"
    if (fileExists(modelFile)) kmeans.load(modelFile)
    else {
      kmeans.maxClusters = 10
      kmeans.train()
    }
    val predictions = kmeans.predict(toCluster.data)
    predictions.show()
    val clusterCenters = kmeans.trainedModel.clusterCenters
    val inertia = kmeans.evaluate(predictions, metricName = "inertia")
    val silhouette = kmeans.evaluate(predictions, metricName = "silhouette")
    Logger.info(s"Cluster centers: ${clusterCenters.mkString(" ")}")
    Logger.info(s"Inertia score: ${inertia(0)}")
    Logger.info(s"Silhouette score: ${silhouette(0)}")

    // Save the model
    if (!fileExists(modelFile)) kmeans.save(modelFile)

    // Stop SparkSession execution
    Spark.stop()
  }

}
