package preprocessing

import org.apache.spark.sql.{DataFrame}

object BoschDataset extends DatasetProperty {

  override val delimiter = ";"

  override def getDiscreteColumnNames(): Array[String] = {
    return Array()
  }

  override def getContinuosColumnNames(): Array[String] = {
    return Array()
  }

  override def getTargetColumnNames(): Array[String] = {
    return Array("Response")
  }

}

class BoschDataset(inputPath: String) extends Dataset(inputPath) {

  override def property = BoschDataset

  override def preprocess(): DataFrame = {
    val funcs = Seq(
      Preprocessor.dropColumns(_: DataFrame, "Id"),
      Preprocessor.binaryConversion(_: DataFrame),
      Preprocessor.pca(_: DataFrame, 50)
    )
    return funcs.foldLeft(data) { (r, f) => f(r) }
  }

}
