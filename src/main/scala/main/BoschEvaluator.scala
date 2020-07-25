package main

import preprocessing.{BoschDataset, Preprocessor}
import prediction.{Predictor, Clusterer}
import utils._

import org.apache.spark.sql.functions.col

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
      .filterNot(c => usefulFeatures.contains(c))

    usefulFeatures.foreach(println)
    notUsefulFeatures.foreach(println)

    // Apply preprocessing for clustering
    val (toCluster, pc) = preprocessed.preprocessForClustering()
    toCluster.show()

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

    // Define classifiers based on the clustering output
    val splittedData = Utils.splitDataFrame(predictions, kmeans.predictionCol)
    val classifiers: Map[Int, Predictor[_]] =
      splittedData.map({
        case (v, d) => {
          var newData = dataset.data.drop(notUsefulFeatures: _*)
          var toClassify = BoschDataset(inputData = Some(newData)).preprocess()
          toClassify.data = toClassify.data.filter(
            !col("Id").isin(Utils.colToArrayOfDouble(d, "Id"): _*)
          )
          (
            v.asInstanceOf[Int],
            Predictor(classifierName.orNull.toString, toClassify)
          )
        }
      })

    // Train each classifier and print results
    classifiers.foreach({
      case (v, c) => {
        c.train(assemble = false)
        var predictions = c.test()
        predictions.show()
        var accuracy = c.evaluate(predictions, metricName = "mcc")
        var mcc = c.evaluate(predictions, metricName = "mcc")
        var auroc = c.evaluate(predictions, metricName = "areaUnderRoc")
        Logger.info(s"Accuracy score for cluster #${v}: ${accuracy}")
        Logger.info(s"MCC score for cluster #${v}: ${mcc}")
        Logger.info(s"Area under ROC score for cluster #${v}: ${auroc}")
      }
    })

    // Stop SparkSession execution
    Spark.stop()
  }

}
