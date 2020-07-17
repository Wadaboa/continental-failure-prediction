package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexer,
  VectorAssembler
}

object Classifier {

  /** Classifier's factory method */
  def apply(name: String, dataset: Dataset): Classifier[_] = {
    name match {
      case "DT" => new DecisionTreeClassifier(dataset)
      case _    => throw new IllegalArgumentException("Unsupported classifier.")
    }
  }

}

abstract class Classifier[M <: PipelineStage](dataset: Dataset) {

  val Array(trainingData, testData) =
    dataset.getData().randomSplit(Array(0.8, 0.2), seed = getRandomSeed())
  val metricName: String = "accuracy"
  val model: M = getModel()
  val pipeline: Pipeline = getPipeline()
  var trainedModel: PipelineModel = _

  /** Defines the model */
  def getModel(): M

  /** Defines model-specific data transformations */
  def getPipeline(): Pipeline

  /** Trains the model */
  def train(): Unit = {
    trainedModel = pipeline.fit(trainingData)
  }

  /** Tests the model */
  def test(): Double = {
    val predictions = trainedModel.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName(metricName)
    return evaluator.evaluate(predictions)
  }

}
