package prediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{DecisionTreeClassifier => DT}


class DecisionTreeClassifier(data: DataFrame) extends Classifier(data) {

  override val model = new DT()
    .setImpurity("entropy")
    .setMaxDepth(10)
    .setSeed(getRandomSeed())

  override def getPipeline(): Pipeline = {
    // Index columns and labels
    val labelIndexer = new StringIndexer()
        .setInputCols(Array(
            "workclass", 
            "education", 
            "marital-status", 
            "occupation", 
            "relationship", 
            "race", 
            "sex", 
            "native-country", 
            "high-income"
        ))
        .setOutputCols(Array(
            "indexed-workclass",
            "indexed-education",
            "indexed-marital-status",
            "indexed-occupation",
            "indexed-relationship",
            "indexed-race",
            "indexed-sex",
            "indexed-native-country",
            "indexed-label"
        ))
        .fit(data)
    
    // Put every feature into a single vector
    val vecAssembler = new VectorAssembler()
        .setInputCols(Array("indexed-workclass",
            "indexed-education",
            "indexed-marital-status",
            "indexed-occupation",
            "indexed-relationship",
            "indexed-race",
            "indexed-sex",
            "indexed-native-country",
            "indexed-label"
        ))
        .setOutputCol("features")

    // Convert index labels back to original labels
    val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predicted-label")
        .setLabels(labelIndexer.labels)

    // Define the pipeline
    val pipeline = new Pipeline()
        .setStages(Array(
            labelIndexer, 
            vecAssembler, 
            model, 
            labelConverter
        ))
    
    return pipeline
  }
  
}
