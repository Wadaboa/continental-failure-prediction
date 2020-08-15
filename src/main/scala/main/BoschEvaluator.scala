package main

import preprocessing.BoschDataset
import prediction.{Predictor, Clusterer}
import utils._

import org.apache.spark.ml.linalg.DenseMatrix

object BoschEvaluator {

  def main(args: Array[String]): Unit = {
    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")
    val classifierName = options.get("classifierName")
    val modelFolder = options.get("modelFolder")

    // Create Dataset object (and read data)
    val dataset = BoschDataset(inputPath = inputPath)
    dataset.show()

    // Apply common preprocessing
    val preprocessed = dataset.preprocessCommon()
    preprocessed.show()

    // Apply preprocessing for clustering
    val (toCluster, clustersPc) = preprocessed.preprocessForClustering()
    toCluster.show()

    // Cluster data and print statistics
    val kmeans = Clusterer("KM", toCluster)
    val clustererModelFile = s"${modelFolder.orNull.toString}/bosch-kmeans.ml"
    if (fileExists(clustererModelFile)) kmeans.load(clustererModelFile)
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

    // Save the clusterer model
    if (!fileExists(clustererModelFile)) kmeans.save(clustererModelFile)

    // Define classifiers based on the clustering output
    val splittedData = Utils.splitDataFrame(predictions, kmeans.predictionCol)
    var classifiersPc: Map[Int, DenseMatrix] = Map()
    val classifiers: Map[Int, Predictor[_]] =
      splittedData.map({
        case (v, d) => {
          Logger.info(s"Preprocessing data for cluster #${v}")
          var newData = preprocessed.data
            .join(d.select("Id"), Seq("Id"), "inner")
          var (toClassify, pc) = BoschDataset(inputData = Some(newData))
            .preprocessForClassification()
          Logger.info(s"Showing data for cluster #${v}")
          toClassify.show()
          classifiersPc += (v.asInstanceOf[Int] -> pc)
          (
            v.asInstanceOf[Int],
            Predictor(classifierName.orNull.toString, toClassify)
          )
        }
      })

    // Train each classifier and print results
    classifiers.foreach({
      case (v, c) => {
        Logger.info(
          s"Training the classifier associated with cluster #${v}"
        )
        var classifierModelFile =
          s"${modelFolder.orNull.toString}/bosch-${classifierName.orNull.toString}-${v}.ml"
        if (fileExists(classifierModelFile)) c.load(classifierModelFile)
        else c.train(assemble = false)
        var predictions = c.test()
        Logger.info(
          s"Showing predictions for the classifier associated with cluster #${v}"
        )
        predictions.show()
        var accuracy = c.evaluate(predictions, metricName = "accuracy")
        var fscore = c.evaluate(predictions, metricName = "f1")
        var mcc = c.evaluate(predictions, metricName = "mcc")
        var auroc = c.evaluate(predictions, metricName = "areaUnderRoc")
        Logger.info(s"Accuracy score for cluster #${v}: ${accuracy}")
        Logger.info(s"F1 score for cluster #${v}: ${fscore}")
        Logger.info(s"MCC score for cluster #${v}: ${mcc}")
        Logger.info(s"Area under ROC score for cluster #${v}: ${auroc}")

        // Save the classifiers models
        if (!fileExists(classifierModelFile)) c.save(classifierModelFile)
      }
    })

    // Stop SparkSession execution
    Spark.stop()
  }

}
