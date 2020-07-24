package main

import preprocessing.ArrestDataset
import prediction.{Predictor, Clusterer}
import utils._

object ArrestEvaluator {

  def main(args: Array[String]): Unit = {
    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")

    // Create Dataset object (and read data)
    val dataset = ArrestDataset(inputPath = inputPath)
    dataset.show()

    // Preprocess data
    val toCluster = dataset.preprocess()
    toCluster.show()

    // Cluster data and print statistics
    val kmeans = Clusterer("KM", toCluster)
    kmeans.maxClusters = 10
    kmeans.train()
    val clusterCenters = kmeans.trainedModel.clusterCenters
    val predictions = kmeans.trainedModel.transform(toCluster.data)
    val inertia = kmeans.evaluate(predictions, metricName = "inertia")
    val silhouette = kmeans.evaluate(predictions, metricName = "silhouette")
    Holder.log.info(s"Cluster centers: ${clusterCenters.mkString(" ")}")
    Holder.log.info(s"Inertia score: ${inertia(0)}")
    Holder.log.info(s"Silhouette score: ${silhouette(0)}")

    // Stop SparkSession execution
    stopSpark()
  }

}
