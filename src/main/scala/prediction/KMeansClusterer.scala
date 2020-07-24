package prediction

import preprocessing.Dataset
import evaluation.{EuclideanInertia, EuclideanGap}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.{Evaluator, ClusteringEvaluator}
import org.apache.spark.ml.clustering.{KMeans => KM, KMeansModel => KMM}

case class KMeansClusterer(dataset: Dataset) extends Clusterer(dataset) {

  override type M = KMM
  override type T = KM

  override var metricName: String = "silhouette"
  var distanceMeasure: String = "euclidean"

  override var model: KM = getNewModel(maxClusters)

  /** Returns a KMeans instance with the specified number of clusters */
  def getNewModel(k: Int) =
    new KM()
      .setK(k)
      .setDistanceMeasure(distanceMeasure)
      .setSeed(getRandomSeed())
      .setFeaturesCol(featuresCol)
      .setPredictionCol(predictionCol)

  override def train(): Unit = {
    model = getNewModel(minClusters)
    trainedModel = model.fit(dataset.data)
    var gapResult = evaluate(
      trainedModel.transform(dataset.data),
      metricName = "gap"
    )
    var previousGap: Double = gapResult(0)
    var previousModel: KM = model
    var previousTrainedModel: KMM = trainedModel
    for (k <- minClusters + 1 to maxClusters) {
      println(s"Num clusters: ${k}")
      model = getNewModel(k)
      trainedModel = model.fit(dataset.data)
      gapResult = evaluate(
        trainedModel.transform(dataset.data),
        metricName = "gap"
      )
      var (gap, stdDev) = (gapResult(0), gapResult(1))
      println(s"Prev gap: ${previousGap}")
      println(s"New gap: ${gap}")
      println(s"New stdev: ${stdDev}")
      if (previousGap > gap - stdDev) {
        println("FOUND BEST")
        model = previousModel
        trainedModel = previousTrainedModel
        return
      }
      previousGap = gap
      previousModel = model
      previousTrainedModel = trainedModel
    }
  }

  override def evaluate(
      predictions: DataFrame,
      metricName: String = this.metricName
  ): Array[Double] = {
    metricName match {
      case "silhouette" =>
        Array(
          new ClusteringEvaluator()
            .setFeaturesCol(featuresCol)
            .setPredictionCol(predictionCol)
            .setMetricName(metricName)
            .evaluate(predictions)
        )
      case "inertia" =>
        Array(
          EuclideanInertia.computeInertiaScore(
            predictions,
            featuresCol,
            trainedModel.clusterCenters
          )
        )
      case "gap" =>
        tuple2ToArray[Double](
          EuclideanGap
            .computeGapScore(
              predictions,
              featuresCol,
              predictionCol,
              trainedModel.clusterCenters
            )
        )
    }
  }

}
