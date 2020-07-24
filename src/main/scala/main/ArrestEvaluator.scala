package main

import preprocessing.ArrestDataset
import prediction.{Predictor, Clusterer}

object ArrestEvaluator {

  def main(args: Array[String]): Unit = {
    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")

    // Create Dataset object (and read data)
    val dataset = ArrestDataset(inputPath = inputPath)
    dataset.show()
    dataset.data.printSchema()

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
    val gap = kmeans.evaluate(predictions, metricName = "gap")
    println(s"Best number of clusters (by gap): ${clusterCenters}")
    println(s"Inertia score: ${inertia}")
    println(s"Silhouette score: ${silhouette}")
    println(s"Gap score: ${gap(0)}")
    println(s"Gap score, standard deviation: ${gap(1)}")

    // Stop SparkSession execution
    stopSpark()
  }

}
