package prediction

import preprocessing.{Dataset}
import utils._

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.evaluation.{Evaluator}

object Clusterer {

  /** Clusterer's factory method */
  def apply(
      name: String,
      dataset: Dataset
  ) =
    name.toUpperCase match {
      case "KM" => KMeansClusterer(dataset)
      case _    => throw new IllegalArgumentException("Unsupported clusterer.")
    }

}

abstract class Clusterer(dataset: Dataset) {

  // Define type variables
  type M <: Model[M]
  type T <: Estimator[M]

  // Define common variables
  var featuresCol: String = "features"
  var predictionCol: String = "prediction"
  var metricName: String
  var numClusters: Option[Int] = None
  var minClusters: Int = 2
  var maxClusters: Int = dataset.getNumRows()

  var model: T
  var trainedModel: M = _

  /** Trains the model */
  def train(): Unit = {
    trainedModel = model.fit(dataset.data)
  }

  /** Predicts on unseen data */
  def predict(data: DataFrame): DataFrame = {
    return trainedModel.transform(data)
  }

  /** Evaluates predictions by returning an array of measurements.
    * The first value in the array is always the requested measure.
    */
  def evaluate(
      predictions: DataFrame,
      metricName: String = this.metricName
  ): Array[Double]

  /** Saves the trained model to disk */
  def save(path: String): Unit

  /** Loads the trained model from disk */
  def load(path: String): Unit

}
