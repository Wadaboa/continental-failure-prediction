package prediction

import preprocessing.{Dataset}
import evaluation.MCC
import utils._

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel, Transformer}
import org.apache.spark.ml.feature.{
  StringIndexer,
  VectorAssembler,
  IndexToString
}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.ml.evaluation.Evaluator

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
  var cvFolds: Int = 3
  var cvConcurrency: Int = 3

  val Array(trainingData, testData) =
    dataset.data.randomSplit(Array(0.8, 0.2), seed = Utils.seed)
  var model: T
  var trainedModel: Transformer = _

  /** Returns the model */
  def getModel(): T = model

  /** Defines model-specific data transformations */
  def pipeline(assemble: Boolean = true): Pipeline = {
    // Get column and target names
    val target = dataset.property.getTargetColumnNames()(0)
    val cols = dataset.getColumnNames().filter(c => c != target)

    // Index labels
    val labelIndexer = new StringIndexer()
      .setInputCol(target)
      .setOutputCol(labelCol)
      .fit(dataset.data)

    // Convert index labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol(predictionCol)
      .setOutputCol("predicted-" + target)
      .setLabels(labelIndexer.labels)

    if (assemble) {
      // Index columns
      val columnsIndexer = new StringIndexer()
        .setInputCols(cols)
        .setOutputCols(cols.map("indexed-" + _))
        .fit(dataset.data)

      // Put every feature into a single vector
      val featuresAssembler = new VectorAssembler()
        .setInputCols(columnsIndexer.getOutputCols)
        .setOutputCol(featuresCol)

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

    return new Pipeline().setStages(
      Array(labelIndexer, getModel(), labelConverter)
    )
  }

  /** Defines the parameter grid to use in cross-validation */
  def paramGrid: Array[ParamMap]

  /** Defines the default evaluator */
  def defaultEvaluator: Evaluator

  /** Trains the model */
  def train(assemble: Boolean = true, validate: Boolean = false): Unit = {
    if (!validate) trainedModel = pipeline(assemble).fit(trainingData)
    else {
      Logger.info(
        s"Estimator param map: ${paramGrid.mkString("[", ", ", "]")}"
      )
      trainedModel = new CrossValidator()
        .setEstimator(pipeline(assemble))
        .setEvaluator(defaultEvaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(cvFolds)
        .setParallelism(cvConcurrency)
        .setSeed(Utils.seed)
        .fit(trainingData)
    }
  }

  /** Tests the model */
  def test(): DataFrame = {
    return predict(testData)
  }

  /** Predicts on unseen data */
  def predict(data: DataFrame): DataFrame = {
    return trainedModel.transform(data)
  }

  /** Evaluates predictions */
  def evaluate(
      predictions: DataFrame,
      metricName: String = this.metricName
  ): Double = {
    metricName match {
      case "mcc" => MCC.computeMccScore(predictions, predictionCol, labelCol)
      case _     => defaultEvaluator.evaluate(predictions)
    }
  }

  /** Saves the trained model to disk */
  def save(path: String): Unit = {
    Logger.info(s"Saving model to ${path}")
    trainedModel match {
      case c: CrossValidatorModel =>
        trainedModel
          .asInstanceOf[CrossValidatorModel]
          .bestModel
          .asInstanceOf[PipelineModel]
          .save(path)
      case p: PipelineModel =>
        trainedModel.asInstanceOf[PipelineModel].save(path)
    }
  }

  /** Loads the trained model from disk */
  def load(path: String): Unit = {
    Logger.info(s"Loading model from ${path}")
    trainedModel = PipelineModel.load(path).asInstanceOf[Transformer]
  }

}
