package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.evaluation.{Evaluator}
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexer,
  VectorAssembler
}

object Predictor {

  /** Predictor's factory method */
  def apply(
      name: String,
      dataset: Dataset,
      split: Boolean = true
  ): Predictor[_] = {
    name match {
      case "DT" => new DecisionTreeClassifier(dataset, split)
      case _    => throw new IllegalArgumentException("Unsupported predictor.")
    }
  }

}

abstract class Predictor[M <: PipelineStage](
    dataset: Dataset,
    split: Boolean = true
) {

  if (split) {
    val Array(trainingData, validationData, testData) =
      dataset.data.randomSplit(Array(0.5, 0.3, 0.2), seed = getRandomSeed())
  }
  val model: M = getModel()
  val pipeline: Pipeline = getPipeline()
  var trainedModel: PipelineModel = _
  val metricName: String
  val evaluator: Evaluator = getEvaluator()

  /** Defines the model */
  def getModel(): M

  /** Defines model-specific data transformations */
  def getPipeline(): Pipeline

  /** Trains the model */
  def train(): Unit = {
    var inputData = dataset.data.select("*")
    if (split) inputData = trainingData
    trainedModel = pipeline.fit(inputData)
  }

  /** Tests the model */
  def test(): DataFrame = {
    var inputData = dataset.data.select("*")
    if (split) inputData = testData
    return trainedModel.transform(inputData)
  }

  /** Defines the evaluator */
  def getEvaluator(): Evaluator

  /** Evaluates predictions */
  def evaluate(predictions: DataFrame): Double = {
    return evaluator.evaluate(predictions)
  }

}
