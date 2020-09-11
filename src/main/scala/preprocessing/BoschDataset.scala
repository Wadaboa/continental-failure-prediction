package preprocessing

import utils._

import org.apache.spark.sql.{DataFrame}
import org.apache.spark.ml.linalg.DenseMatrix

object BoschDataset extends DatasetProperty {

  override val delimiter = ","

  override def getTargetColumnNames(): Array[String] = {
    return Array("Response")
  }

}

case class BoschDataset(
    inputPath: Option[String] = None,
    inputData: Option[DataFrame] = None,
    cacheData: Boolean = true
) extends Dataset(inputPath, inputData, cacheData) {

  override type T = BoschDataset

  override def property = BoschDataset

  /** Common preprocessing for both clustering and classification steps */
  def preprocessCommon(): BoschDataset = {
    val funcs = Seq(
      Preprocessor.dropNullColumns(_: DataFrame),
      Preprocessor.dropConstantColumns(_: DataFrame)
    )
    return BoschDataset(inputData = Some(funcs.foldLeft(data) { (r, f) =>
      f(r)
    }))
  }

  /** Preprocesses data for clustering */
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

  /** Preprocesses data for classification */
  def preprocessForClassification(): Tuple2[BoschDataset, DenseMatrix] = {
    val x = Preprocessor.dropNullColumns(data)
    val y = Preprocessor.dropConstantColumns(x)
    val w = Preprocessor.nullToValues(
      y,
      method = "mean",
      exclude = Array("Id", "Response")
    )
    val (z, pc) = Preprocessor.pca(
      w,
      maxComponents = 50,
      assembleFeatures = true,
      standardizeFeatures = true,
      explainedVariance = 0.95,
      exclude = Array("Id", "Response")
    )
    return (BoschDataset(inputData = Some(z)), pc)
  }

}
