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

  override var model: KM = new KM()
    .setK(maxClusters)
    .setDistanceMeasure(distanceMeasure)
    .setSeed(getRandomSeed())
    .setFeaturesCol(featuresCol)
    .setPredictionCol(predictionCol)

  override def train(): Unit = {
    var bestSilhouette: Double = -1
    var bestModel: KM = null
    var bestTrainedModel: KMM = null
    for (k <- minClusters to maxClusters) {
      var tempModel: KM = model
      var tempTrainedModel: KMM = tempModel.fit(dataset.data)
      var silhouette = evaluate(
        tempTrainedModel.transform(dataset.data),
        metricName = "silhouette"
      )
      if (silhouette > bestSilhouette) {
        bestModel = tempModel
        bestTrainedModel = tempTrainedModel
        bestSilhouette = silhouette
      }
    }
    model = bestModel
    trainedModel = bestTrainedModel
  }

  override def evaluate(
      predictions: DataFrame,
      metricName: String = this.metricName
  ): Double = {
    metricName match {
      case "silhouette" =>
        new ClusteringEvaluator()
          .setFeaturesCol(featuresCol)
          .setPredictionCol(predictionCol)
          .setMetricName(metricName)
          .evaluate(predictions)
      case "inertia" =>
        EuclideanInertia.computeInertiaScore(
          predictions,
          featuresCol,
          trainedModel.clusterCenters
        )
      case "gap" =>
        EuclideanGap
          .computeGapScore(
            predictions,
            featuresCol,
            trainedModel.clusterCenters
          )
          ._1
    }
  }

}
