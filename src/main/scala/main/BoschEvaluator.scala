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

    import spark.implicits._
    val data = Seq(
      ("Java", "20000", null, 1),
      ("Python", "100000", null, 1),
      ("Scala", "3000", null, 1),
      ("Rust", null, null, 1),
      ("R", null, null, 1)
    )
    val rdd = spark.sparkContext.parallelize(data)
    val df = rdd.toDF()
    df.show()
    val x = Preprocessor.dropNullColumns(df)
    x.show()

    /*
    // Create Dataset object (and read data)
    val dataset = BoschDataset(inputPath = inputPath)
    dataset.show()

    // Preprocess data
    val toCluster = dataset.preprocessForClustering()
    toCluster.show()


    // Cluster data and print centroids
    val kmeans = Clusterer("KM", toCluster)
    kmeans.maxClusters = 3
    kmeans.train()
    kmeans.trainedModel.clusterCenters.foreach(println)
    val predictions = kmeans.trainedModel.transform(dataset.data)
    val silhouette = kmeans.evaluate(predictions, metricName = "silhouette")
    println(silhouette)
    val gap = kmeans.evaluate(predictions, metricName = "gap")
    println(gap)
     */
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
