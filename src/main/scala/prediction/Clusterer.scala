package prediction

import preprocessing.{Dataset}

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
  var minClusters: Int = 2
  var maxClusters: Int = dataset.getNumRows()

  var model: T

  /** Returns the model */
  def getModel(): T = model

  /** Trains the model */
  def train(): M = {
    return model.fit(dataset.data)
  }

  /** Defines the evaluator */
  def evaluator: Evaluator

  /** Evaluates predictions */
  def evaluate(predictions: DataFrame): Double = {
    return evaluator.evaluate(predictions)
  }

}
