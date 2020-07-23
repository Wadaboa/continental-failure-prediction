package prediction

import preprocessing.Dataset
import evaluation.{EuclideanInertia, EuclideanGap}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.{Evaluator, ClusteringEvaluator}
import org.apache.spark.ml.clustering.{KMeans => KM, KMeansModel => KMM}

protected case class KMeansClusterer(dataset: Dataset)
    extends Clusterer(dataset) {

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
    var data = dataset.data
    model = getNewModel(minClusters)
    trainedModel = model.fit(data)
    var gapResult = evaluate(
      trainedModel.transform(data),
      metricName = "gap"
    )
    println(gapResult)
    var previousGap: Double = gapResult(0)
    var previousStdDev: Double = gapResult(1)
    var previousModel: KM = model
    var previousTrainedModel: KMM = trainedModel
    var previousData: DataFrame = data
    for (k <- minClusters + 1 to maxClusters) {
      println(k)
      data = data.drop(predictionCol)
      model = getNewModel(k)
      trainedModel = model.fit(data)
      gapResult = evaluate(
        trainedModel.transform(data),
        metricName = "gap"
      )
      println(gapResult)
      var (gap, stdDev) = (gapResult(0), gapResult(1))
      if (previousGap > gap - stdDev) {
        println("FOUND BEST")
        dataset.data = data
        model = previousModel
        trainedModel = previousTrainedModel
        return
      }
      previousGap = gap
      previousStdDev = stdDev
      previousData = data
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
        EuclideanGap
          .computeGapScore(
            predictions,
            featuresCol,
            predictionCol,
            trainedModel.clusterCenters
          )
          .asInstanceOf[Array[Double]]
    }
  }

}
