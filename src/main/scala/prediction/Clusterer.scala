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
  ): Clusterer = {
    name match {
      case "KM" => new KMeansClusterer(dataset)
      case _    => throw new IllegalArgumentException("Unsupported clusterer.")
    }
  }

}

abstract class Clusterer(dataset: Dataset) {

  // Define type variables
  type M <: Model[M]
  type T <: Estimator[M]

  var minClusters: Int = 2
  var maxClusters: Int = dataset.getNumRows()
  var model: T = getModel()
  val metricName: String = getMetricName()
  val evaluator: Evaluator = getEvaluator()

  /** Defines the model */
  def getModel(): T

  /** Trains the model */
  def train(): M = {
    return model.fit(dataset.data)
  }

  /** Defines the evaluator */
  def getEvaluator(): Evaluator

  /** Defines the metric name */
  def getMetricName(): String

  /** Evaluates predictions */
  def evaluate(predictions: DataFrame): Double = {
    return evaluator.evaluate(predictions)
  }

}
