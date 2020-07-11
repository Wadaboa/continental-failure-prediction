package prediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{DecisionTreeClassifier => DT}


class DecisionTreeClassifier(data: DataFrame) extends Classifier(data) {

  override val model = new DT().setSeed(getRandomSeed())

  override def getPipeline(): Pipeline = {
      return null
  }
  
}
