package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.{Evaluator, ClusteringEvaluator}
import org.apache.spark.ml.clustering.{KMeans => KM, KMeansModel => KMM}

class KMeansClusterer(dataset: Dataset, split: Boolean = false)
    extends Predictor[KM](dataset, split) {

  override val metricName: String = "silhouette"
  val maxClusters: Integer = dataset.getNumRows()

  override def runValidation(): Unit = {
    var bestSilhouette: Double = -1
    var bestModel: KM = null
    var bestTrainedModel: KMM = null
    for (k <- 1 to maxClusters) {
      var tempModel = new KM().setK(k)
      var tempTrainedModel = tempModel.fit(getTrainingData())
      var silhouette = evaluate(tempTrainedModel.transform(getTestData()))
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
