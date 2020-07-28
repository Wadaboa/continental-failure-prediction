package preprocessing

import utils._

import org.apache.spark.sql.{DataFrame}
import org.apache.spark.ml.linalg.DenseMatrix

object BoschDataset extends DatasetProperty {

  override val delimiter = ";"

  override def getTargetColumnNames(): Array[String] = {
    return Array("Response")
  }

}

case class BoschDataset(
    inputPath: Option[String] = None,
    inputData: Option[DataFrame] = None
) extends Dataset(inputPath, inputData) {

  override type T = BoschDataset

  override def property = BoschDataset

  override def preprocess(): BoschDataset = {
    val funcs = Seq(
      Preprocessor
        .pca(
          _: DataFrame,
          maxComponents = 50,
          assembleFeatures = true,
          standardizeFeatures = true,
          explainedVariance = 0.95,
          exclude = Array("Id", "Response")
        )
        ._1
    )
    return new BoschDataset(inputData = Some(funcs.foldLeft(data) { (r, f) =>
      f(r)
    }))
  }

  def preprocessCommon(): BoschDataset = {
    val funcs = Seq(
      Preprocessor.dropNullColumns(_: DataFrame),
      Preprocessor.dropConstantColumns(_: DataFrame)
    )
    return BoschDataset(inputData = Some(funcs.foldLeft(data) { (r, f) =>
      f(r)
    }))
  }

  def preprocessForClustering(): Tuple2[BoschDataset, DenseMatrix] = {
    val x = Preprocessor.binaryConversion(
      data,
      exclude = Array("Id", "Response")
    )
    val (y, pc) = Preprocessor.pca(
      x,
      maxComponents = 50,
      assembleFeatures = true,
      standardizeFeatures = false,
      explainedVariance = 0.95,
      exclude = Array("Id", "Response")
    )
    return (BoschDataset(inputData = Some(y)), pc)
  }

}
