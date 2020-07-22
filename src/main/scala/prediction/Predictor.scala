package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.evaluation.{Evaluator}

object Predictor {

  /** Predictor's factory method */
  def apply(
      name: String,
      dataset: Dataset
  ): Predictor = {
    name match {
      case "DT" => new DecisionTreeClassifier(dataset)
      case _    => throw new IllegalArgumentException("Unsupported predictor.")
    }
  }

}

abstract class Predictor(dataset: Dataset) {

  // Define type variables
  type M <: PipelineStage

  val Array(trainingData, validationData, testData) =
    dataset.data.randomSplit(Array(0.5, 0.3, 0.2), seed = getRandomSeed())
  var model: M = getModel()
  val pipeline: Pipeline = getPipeline()
  var trainedModel: PipelineModel = _
  val metricName: String = getMetricName()
  val evaluator: Evaluator = getEvaluator()

  /** Defines the model */
  def getModel(): M

  /** Defines model-specific data transformations */
  def getPipeline(): Pipeline

  /** Trains the model */
  def train(): Unit = {
    trainedModel = pipeline.fit(trainingData)
  }

  /** Tests the model */
  def test(): DataFrame = {
    return trainedModel.transform(testData)
  }

  /** Validates the model */
  def validate(): DataFrame = {
    return trainedModel.transform(validationData)
  }

  /** Defines the metric name */
  def getMetricName(): String

  /** Defines the evaluator */
  def getEvaluator(): Evaluator

  /** Evaluates predictions */
  def evaluate(predictions: DataFrame): Double = {
    return evaluator.evaluate(predictions)
  }

}
