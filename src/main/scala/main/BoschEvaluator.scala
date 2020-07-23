package main

import preprocessing.Dataset
import prediction.{Predictor, Clusterer}

object BoschEvaluator {

  def main(args: Array[String]): Unit = {
    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")
    val classifierName = options.get("classifierName")

    // Create Dataset object (and read data)
    val dataset = Dataset("Bosch", inputPath = inputPath)
    dataset.show()

    // Preprocess data
    val toCluster = dataset.preprocess()
    toCluster.renameColumn("pcaFeatures", "features")
    toCluster.show()

    // Cluster data and print centroids
    val kmeans = Clusterer("KM", toCluster)
    kmeans.maxClusters = 3
    kmeans.metricName = "inertia"
    kmeans.train()
    kmeans.trainedModel.clusterCenters.foreach(println)
    val predictions = kmeans.trainedModel.transform(dataset.data)
    val result = kmeans.evaluate(predictions)
    println(result)

    /*
    // Train the classifier and test it
    val classifier = Classifier(classifierName.orNull.toString, dataset)
    classifier.train()
    val result = classifier.test()
    println(result)
     */

    // Stop SparkSession execution
    stopSpark()
  }

}
