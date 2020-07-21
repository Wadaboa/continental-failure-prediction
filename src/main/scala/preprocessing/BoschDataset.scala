package preprocessing

import org.apache.spark.sql.{DataFrame}

object BoschDataset extends DatasetProperty {

  override val delimiter = ";"

  override def getTargetColumnNames(): Array[String] = {
    return Array("Response")
  }

}

class BoschDataset(
    inputPath: Option[String] = None,
    var inputData: Option[DataFrame] = None
) extends Dataset(inputPath, inputData) {

  override def property = BoschDataset

  override def preprocess(): DataFrame = { return null }

  override def preprocessForClustering(): DataFrame = {
    val funcs = Seq(
      Preprocessor.dropColumns(_: DataFrame, "Id"),
      Preprocessor.binaryConversion(_: DataFrame),
      Preprocessor.pca(_: DataFrame, 50)
    )
    return funcs.foldLeft(data) { (r, f) => f(r) }
  }

}
