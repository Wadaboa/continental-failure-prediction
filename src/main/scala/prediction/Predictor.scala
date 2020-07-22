package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.feature.{
  StringIndexer,
  VectorAssembler,
  IndexToString
}
import org.apache.spark.ml.evaluation.{Evaluator}

object Predictor {

  /** Predictor's factory method */
  def apply(
      name: String,
      dataset: Dataset
  ): Predictor = {
    name match {
      case "DT" => new DecisionTreeClassifier(dataset)
      case "RF" => new RandomForestClassifier(dataset)
      case _    => throw new IllegalArgumentException("Unsupported predictor.")
    }
  }

}

abstract class Predictor(dataset: Dataset) {

  // Define common variables
  var labelCol: String = "label"
  var featuresCol: String = "features"
  var predictionCol: String = "prediction"
  var metricName: String = "accuracy"

  // Define type variables
  type T <: PipelineStage

  val Array(trainingData, validationData, testData) =
    dataset.data.randomSplit(Array(0.5, 0.3, 0.2), seed = getRandomSeed())
  var model: T = getModel()
  var trainedModel: PipelineModel = _
  val pipeline: Pipeline = getPipeline()
  val evaluator: Evaluator = getEvaluator()

  /** Defines the model */
  def getModel(): T

  /** Defines model-specific data transformations */
  def getPipeline(): Pipeline = {
    // Get column and target names
    val target = dataset.property.getTargetColumnNames()(0)
    val cols = dataset.getColumnNames().filter(c => c != target)

    // Index columns
    val columnsIndexer = new StringIndexer()
      .setInputCols(cols)
      .setOutputCols(cols.map("indexed-" + _))
      .fit(dataset.data)

    // Index labels
    val labelIndexer = new StringIndexer()
      .setInputCol(target)
      .setOutputCol(labelCol)
      .fit(dataset.data)

    // Put every feature into a single vector
    val featuresAssembler = new VectorAssembler()
      .setInputCols(columnsIndexer.getOutputCols)
      .setOutputCol(featuresCol)

    // Convert index labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol(predictionCol)
      .setOutputCol("predicted-label")
      .setLabels(labelIndexer.labels)

    // Define the pipeline
    val pipeline = new Pipeline()
      .setStages(
        Array(
          columnsIndexer,
          labelIndexer,
          featuresAssembler,
          model,
          labelConverter
        )
      )

    return pipeline
  }

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

  /** Defines the evaluator */
  def getEvaluator(): Evaluator

  /** Evaluates predictions */
  def evaluate(predictions: DataFrame): Double = {
    return evaluator.evaluate(predictions)
  }

}
