package main

import preprocessing.BoschDataset
import prediction.{Predictor, Clusterer}
import utils._

import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.storage.StorageLevel

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

    val distinct = Utils.distinctValuesCount(dataset.data, "Response")
    Logger.info("Number of rows for each distinct value in the target column:")
    distinct.show()

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

    // Define classifiers based on the clustering output and train them
    Logger.info(
      s"Number of partitions for common preprocessed data: ${preprocessed.data.rdd.getNumPartitions}"
    )
    val splittedData = Utils.splitDataFrame(predictions, kmeans.predictionCol)
    val bSplittedData = Spark.context.broadcast(splittedData)
    var classifiersPc: Map[Int, DenseMatrix] = Map()
    bSplittedData.value.foreach({
      case (v, d) => {
        // Preprocess clustered data for classification
        Logger.info(s"Preprocessing data for cluster #${v}")
        Logger.info(
          s"Number of partitions in predictions for cluster #${v}: ${d.rdd.getNumPartitions}"
        )
        var newData =
          preprocessed.data.join(d.select("Id"), Seq("Id"), "inner")
        Logger.info(
          s"Number of partitions in joined data for cluster #${v}: ${newData.rdd.getNumPartitions}"
        )
        Logger.info(s"Showing joined data for cluster #${v}")
        newData.show()
        var (toClassify, pc) = BoschDataset(inputData = Some(newData))
          .preprocessForClassification()
        Logger.info(
          s"Cluster #${v} data shape: (${toClassify.getNumRows()}, ${toClassify.getNumCols()})"
        )
        Logger.info(s"Showing data for cluster #${v}")
        toClassify.show()
        val distinct = Utils.distinctValuesCount(toClassify.data, "Response")
        Logger.info(
          s"Number of rows for each distinct value in the target column for cluster #${v}:"
        )
        distinct.show()
        classifiersPc += (v.asInstanceOf[Int] -> pc)

        // Train each classifier and print results
        var classifier = Predictor(classifierName.orNull.toString, toClassify)
        Logger.info(s"Training cluster #${v} classifier")
        var classifierModelFile =
          s"${modelFolder.orNull.toString}/bosch-${classifierName.orNull.toString}-${v}.ml"
        if (fileExists(classifierModelFile))
          classifier.load(classifierModelFile)
        else classifier.train(assemble = false, validate = true)
        var predictions = classifier.test()
        Logger.info(s"Showing cluster #${v} classifier predictions")
        predictions.show()

        // Compute and print evaluations
        var accuracy = classifier.evaluate(predictions, metric = "accuracy")
        var precision = classifier.evaluate(predictions, metric = "precision")
        var recall = classifier.evaluate(predictions, metric = "recall")
        var fscore = classifier.evaluate(predictions, metric = "f1")
        var mcc = classifier.evaluate(predictions, metric = "mcc")
        var auroc = classifier.evaluate(predictions, metric = "areaUnderROC")
        Logger.info(s"Scores for cluster #${v} classifier")
        Logger.info(s"Accuracy score: ${accuracy}")
        Logger.info(s"Precision score: ${precision}")
        Logger.info(s"Recall score: ${recall}")
        Logger.info(s"F1 score: ${fscore}")
        Logger.info(s"Area under ROC score: ${auroc}")
        Logger.info(s"MCC score: ${mcc}")

        // Save the classifiers models
        if (!fileExists(classifierModelFile))
          classifier.save(classifierModelFile)
      }
    })

    // Stop SparkSession execution
    Spark.stop()
  }

}
