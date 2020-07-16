package prediction

import preprocessing.{Dataset, DatasetProperty}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.classification.{DecisionTreeClassifier => DT}

class DecisionTreeClassifier(
    dataset: Dataset,
    datasetProperty: DatasetProperty
) extends Classifier[DT](dataset, datasetProperty) {

  override def getModel(): DT = {
    return new DT()
      .setImpurity("entropy")
      .setMaxDepth(10)
      .setSeed(getRandomSeed())
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
  }

  override def getPipeline(): Pipeline = {
    // Columns to index to numbers
    val toIndex = Array(
      "workclass",
      "education",
      "marital-status",
      "occupation",
      "relationship",
      "race",
      "sex",
      "native-country"
    )

    // Index columns
    val columnsIndexer = new StringIndexer()
      .setInputCols(toIndex)
      .setOutputCols(toIndex.map("indexed" + _))
      .fit(dataset.getData())

    // Index labels
    val labelIndexer = new StringIndexer()
      .setInputCol(datasetProperty.getTargetColumnNames()(0))
      .setOutputCol("label")
      .fit(dataset.getData())

    // Put every feature into a single vector
    val featuresAssembler = new VectorAssembler()
      .setInputCols(columnsIndexer.getOutputCols)
      .setOutputCol("features")

    // Convert index labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol(model.getPredictionCol)
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

}
