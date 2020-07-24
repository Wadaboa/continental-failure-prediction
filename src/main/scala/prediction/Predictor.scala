package prediction

import preprocessing.{Dataset}
import utils._

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, Transformer}
import org.apache.spark.ml.feature.{
  StringIndexer,
  VectorAssembler,
  IndexToString
}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.{Evaluator}

object Predictor {

  /** Predictor's factory method */
  def apply(
      name: String,
      dataset: Dataset
  ) =
    name.toUpperCase match {
      case "DT"  => DecisionTreeClassifier(dataset)
      case "RF"  => RandomForestClassifier(dataset)
      case "GBT" => GradientBoostedClassifier(dataset)
      case _     => throw new IllegalArgumentException("Unsupported predictor.")
    }

}

abstract class Predictor[T <: PipelineStage](dataset: Dataset) {

  // Define common variables
  var labelCol: String = "label"
  var featuresCol: String = "features"
  var predictionCol: String = "prediction"
  var metricName: String = "accuracy"
  var cvFolds: Int = 5
  var cvConcurrency: Int = 2

  val Array(trainingData, testData) =
    dataset.data.randomSplit(Array(0.8, 0.2), seed = Utils.seed)
  var model: T
  var trainedModel: Transformer = _

  /** Returns the model */
  def getModel(): T = model

  /** Defines model-specific data transformations */
  def pipeline: Pipeline = {
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
      .setOutputCol("predicted-" + target)
      .setLabels(labelIndexer.labels)

    // Define the pipeline
    return new Pipeline()
      .setStages(
        Array(
          columnsIndexer,
          labelIndexer,
          featuresAssembler,
          getModel(),
          labelConverter
        )
      )
  }

  /** Defines the parameter grid to use in cross-validation */
  def paramGrid: Array[ParamMap]

  /** Trains the model */
  def train(validate: Boolean = false): Unit = {
    if (!validate) trainedModel = pipeline.fit(trainingData)
    else {
      trainedModel = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(cvFolds)
        .setParallelism(cvConcurrency)
        .setSeed(Utils.seed)
        .fit(trainingData)
    }
  }

  /** Tests the model */
  def test(): DataFrame = {
    return trainedModel
      .transform(testData)
      .select(labelCol, predictionCol)
  }

  /** Defines the evaluator */
  def evaluator: Evaluator

  /** Evaluates predictions */
  def evaluate(predictions: DataFrame): Double = {
    return evaluator.evaluate(predictions)
  }

}
