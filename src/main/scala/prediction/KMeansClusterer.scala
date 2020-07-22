package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.{Evaluator, ClusteringEvaluator}
import org.apache.spark.ml.clustering.{KMeans => KM, KMeansModel => KMM}

class KMeansClusterer(dataset: Dataset) extends Clusterer(dataset) {

  override type M = KMM
  override type T = KM

  override def train(): KMM = {
    var bestSilhouette: Double = -1
    var bestModel: KM = null
    var bestTrainedModel: KMM = null
    for (k <- minClusters to maxClusters) {
      var tempModel: KM = new KM().setK(k)
      var tempTrainedModel: KMM = tempModel.fit(dataset.data)
      var silhouette = evaluate(tempTrainedModel.transform(dataset.data))
      println(silhouette)
      if (silhouette > bestSilhouette) {
        println("NEW BEST")
        bestModel = tempModel
        bestTrainedModel = tempTrainedModel
        bestSilhouette = silhouette
      }
    }
    model = bestModel
    return bestTrainedModel
  }

  override def getModel(): KM = {
    return new KM()
      .setK(maxClusters)
      .setSeed(getRandomSeed())
  }

  override def getMetricName(): String = "silhouette"

  override def getEvaluator(): Evaluator = {
    return new ClusteringEvaluator()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMetricName(metricName)
  }

}
