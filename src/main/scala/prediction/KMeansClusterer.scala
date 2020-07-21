package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.{Evaluator, ClusteringEvaluator}
import org.apache.spark.ml.clustering.{KMeans => KM, KMeansModel => KMM}

class KMeansClusterer(dataset: Dataset) extends Clusterer[KM, KMM](dataset) {

  override val metricName: String = "silhouette"

  override def train(): Unit = {
    var bestSilhouette: Double = -1
    var bestModel: KM = null
    var bestTrainedModel: KMM = null
    for (k <- 1 to maxClusters) {
      var tempModel = new KM().setK(k)
      var tempTrainedModel = tempModel.fit(dataset.data)
      var silhouette = evaluate(tempTrainedModel.transform(dataset.data))
      if (silhouette > bestSilhouette) {
        bestModel = tempModel
        bestTrainedModel = tempTrainedModel
        bestSilhouette = silhouette
      }
    }
    model = bestModel
    trainedModel = bestTrainedModel
  }

  override def getModel(): KM = {
    return new KM()
      .setK(maxClusters)
      .setSeed(getRandomSeed())
  }

  def getEvaluator(): Evaluator = {
    return new ClusteringEvaluator()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMetricName(metricName)
  }

}
