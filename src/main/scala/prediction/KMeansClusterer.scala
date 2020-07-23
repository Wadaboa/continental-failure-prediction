package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.{Evaluator, ClusteringEvaluator}
import org.apache.spark.ml.clustering.{KMeans => KM, KMeansModel => KMM}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.clustering.DistanceMeasure
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}

protected case class KMeansClusterer(dataset: Dataset)
    extends Clusterer(dataset) {

  override type M = KMM
  override type T = KM

  override var metricName: String = "silhouette"
  val distanceMeasure: String = "euclidean"

  override var model: KM = new KM()
    .setK(maxClusters)
    .setDistanceMeasure(distanceMeasure)
    .setSeed(getRandomSeed())
    .setFeaturesCol(featuresCol)
    .setPredictionCol(predictionCol)

  override def train(): KMM = {
    var bestSilhouette: Double = -1
    var bestModel: KM = null
    var bestTrainedModel: KMM = null
    for (k <- minClusters to maxClusters) {
      var tempModel: KM = new KM().setK(k)
      var tempTrainedModel: KMM = tempModel.fit(dataset.data)
      var silhouette = evaluate(tempTrainedModel.transform(dataset.data))
      if (silhouette > bestSilhouette) {
        bestModel = tempModel
        bestTrainedModel = tempTrainedModel
        bestSilhouette = silhouette
      }
    }
    model = bestModel
    return bestTrainedModel
  }

  override def evaluator: Evaluator =
    new ClusteringEvaluator()
      .setFeaturesCol(featuresCol)
      .setPredictionCol(predictionCol)
      .setMetricName(metricName)

  /** Compute the Within Set Sum of Squared Errors */
  def inertia(trainedModel: KMM): Double = {
    val bClusterCenters = dataset.data.sparkSession.sparkContext.broadcast(
      trainedModel.clusterCenters.map(p =>
        new VectorWithNorm(OldVectors.fromML(p))
      )
    )
    val cost = dataset.data
      .select(featuresCol)
      .rdd
      .map { row =>
        pointCost(
          bClusterCenters.value,
          new VectorWithNorm(OldVectors.fromML(row.getAs[Vector](0)))
        )
      }
      .sum()
    bClusterCenters.destroy()
    cost
  }

}
