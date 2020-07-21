package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.evaluation.{Evaluator}

object Clusterer {

  /** Clusterer's factory method */
  def apply(
      name: String,
      dataset: Dataset
  ): Clusterer[_, _] = {
    name match {
      case "KM" => new KMeansClusterer(dataset)
      case _    => throw new IllegalArgumentException("Unsupported clusterer.")
    }
  }

}

abstract class Clusterer[T <: Estimator[M], M <: Model[M]](dataset: Dataset) {

  var model: T = getModel()
  var trainedModel: M = _
  val metricName: String
  val evaluator: Evaluator = getEvaluator()

  /** Defines the model */
  def getModel(): T

  /** Trains the model */
  def train(): Unit = {
    trainedModel = model.fit(dataset.data)
  }

  /** Defines the evaluator */
  def getEvaluator(): Evaluator

  /** Evaluates predictions */
  def evaluate(predictions: DataFrame): Double = {
    return evaluator.evaluate(predictions)
  }

}
