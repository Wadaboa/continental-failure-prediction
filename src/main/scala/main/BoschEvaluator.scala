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

    // Apply common preprocessing
    val preprocessed = dataset.preprocessCommon()
    preprocessed.show()
    val usefulFeatures = preprocessed.getColumnNames()
    val notUsefulFeatures = dataset
      .getColumnNames()
      .filterNot(c =>
        usefulFeatures
          .contains(c) || dataset.property.getTargetColumnNames().contains(c)
      )

    usefulFeatures.foreach(println)

    // Apply preprocessing for clustering
    val (toCluster, pc) = preprocessed.preprocessForClustering()
    toCluster.show()

    toCluster.data = toCluster.data.drop("Id")

    // Cluster data and print statistics
    val kmeans = Clusterer("KM", toCluster)
    kmeans.maxClusters = 10
    kmeans.train()
    val predictions = kmeans.predict(toCluster.data)
    predictions.show()
    val clusterCenters = kmeans.trainedModel.clusterCenters
    val numClusters = clusterCenters.length
    val inertia = kmeans.evaluate(predictions, metricName = "inertia")
    val silhouette = kmeans.evaluate(predictions, metricName = "silhouette")
    Logger.info(s"Cluster centers: ${clusterCenters.mkString(" ")}")
    Logger.info(s"Inertia score: ${inertia(0)}")
    Logger.info(s"Silhouette score: ${silhouette(0)}")

    // Train different classifiers based on the clustering output
    val splittedData = Utils.splitDataframe(predictions, kmeans.predictionCol)
    val classifiers =
      splittedData.map({
        case (v, d) => {
          var newData = dataset.data.drop(notUsefulFeatures)
          newData = Preprocessor.assemble(toClassify.data, "features")
          newData = Preprocessor.toComponents(newData, pc, "features")
          val toClassify = new BoschDataset(inputData = newData)
          (v, Predictor(classifierName.orNull.toString, d))
        }

      })

    // Stop SparkSession execution
    Spark.stop()
  }

}
