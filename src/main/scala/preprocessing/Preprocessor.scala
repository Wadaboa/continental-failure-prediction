package preprocessing

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.DataFrame

object Preprocessor {

  def preprocess(data: DataFrame): Unit = {

    println("Nothing")
  
  }
  
}
