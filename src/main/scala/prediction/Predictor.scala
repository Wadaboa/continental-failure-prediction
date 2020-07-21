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
      case "KM" => new KMeansClusterer(dataset, split)
      case _    => throw new IllegalArgumentException("Unsupported predictor.")
    }
  }

}

abstract class Predictor[M <: PipelineStage](
    dataset: Dataset,
    split: Boolean = true
) {

  val Array(trainingData, validationData, testData) =
    dataset.data.randomSplit(Array(0.5, 0.3, 0.2), seed = getRandomSeed())
  var model: M = getModel()
  val pipeline: Pipeline = getPipeline()
  var trainedModel: PipelineModel = _
  val metricName: String
  val evaluator: Evaluator = getEvaluator()

  /** Defines the model */
  def getModel(): M

  /** Defines model-specific data transformations */
  def getPipeline(): Pipeline

  /** Returns the correct training set */
  def getTrainingData(): DataFrame = {
    if (split) return trainingData
    return dataset.data
  }

  /** Trains the model */
  def train(): Unit = {
    trainedModel = pipeline.fit(getTrainingData())
  }

  /** Returns the correct test set */
  def getTestData(): DataFrame = {
    if (split) return testData
    return dataset.data
  }

  /** Tests the model */
  def test(): DataFrame = {
    return trainedModel.transform(getTestData())
  }

  /** Returns the correct validation set */
  def getValidationData(): DataFrame = {
    if (split) return validationData
    return dataset.data
  }

  /** Validates the model */
  def validate(): DataFrame = {
    return trainedModel.transform(getValidationData())
  }

  /** Finds the best hyper-parameters */
  def runValidation(): Unit

  /** Defines the evaluator */
  def getEvaluator(): Evaluator

  /** Evaluates predictions */
  def evaluate(predictions: DataFrame): Double = {
    return evaluator.evaluate(predictions)
  }

}
