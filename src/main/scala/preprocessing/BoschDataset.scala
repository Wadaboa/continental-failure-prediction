package preprocessing

import utils._

import org.apache.spark.sql.{DataFrame}

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
      Preprocessor.dropColumns(_: DataFrame, "Id"),
      Preprocessor.binaryConversion(_: DataFrame),
      Preprocessor.pca(
        _: DataFrame,
        maxComponents = 50,
        explainedVariance = 0.60
      )
    )
    return new BoschDataset(inputData = Some(funcs.foldLeft(data) { (r, f) =>
      f(r)
    }))
  }

  def preprocessForClustering(): BoschDataset = {
    val funcs = Seq(
      Preprocessor.takeSubset(_: DataFrame, p = 0.5),
      Preprocessor.dropColumns(_: DataFrame, "Id", "Response"),
      Preprocessor.dropNullColumns(_: DataFrame),
      Preprocessor.dropConstantColumns(_: DataFrame),
      Preprocessor.binaryConversion(_: DataFrame),
      Preprocessor.pca(
        _: DataFrame,
        maxComponents = 50,
        explainedVariance = 0.95
      ),
      Preprocessor.fromVectorToDataframe(
        _: DataFrame,
        "features",
        maintainVector = true
      )
    )
    return new BoschDataset(inputData = Some(funcs.foldLeft(data) { (r, f) =>
      f(r)
    }))
  }

}
