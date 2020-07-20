package preprocessing

import org.apache.spark.sql.{SparkSession, DataFrame, DataFrameReader}
import org.apache.spark.sql.types.StructType

trait DatasetProperty {

  /** CSV dataset delimiter */
  val delimiter = ","

  /** Specifies discrete columns */
  def getDiscreteColumnNames(): Array[String] = Array()

  /** Specifies continuos columns */
  def getContinuosColumnNames(): Array[String] = Array()

  /** Specifies target columns */
  def getTargetColumnNames(): Array[String] = Array()

  /** Get useful column names */
  def getColumnNames(): Array[String] = {
    return getDiscreteColumnNames() ++ getContinuosColumnNames() ++ getTargetColumnNames()
  }

  /** Returns the dataset schema string */
  def getSchemaString(): String = null

  /** Returns the dataset schema */
  def getSchema(): StructType = {
    val schemaString = getSchemaString()
    if (schemaString != null) return StructType.fromDDL(getSchemaString())
    return null
  }

  /** Loads the dataset */
  def load(inputPath: String): DataFrame = {
    // Prepare the DataFrameReader
    var reader: DataFrameReader = SparkSession.builder
      .getOrCreate()
      .read

    // Check if a schema is provided (otherwise try to infer it)
    val schema = getSchema()
    if (schema != null) reader = reader.schema(schema)
    else reader = reader.option("inferSchema", "true")

    // Load the dataset
    return reader
      .format("csv")
      .option("delimiter", delimiter)
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load(inputPath)
  }

}

object Dataset {

  def apply(inputPath: String): Dataset = {
    val dataset = new Dataset(Some(inputPath))
    dataset.data = dataset.property.load(inputPath.orNull.toString)
  }

  def apply(data: Option[DataFrame]): Dataset = {
    val dataset = new Dataset
    dataset.data = data
    return dataset
  }

}

class Dataset private (data: DataFrame) {

  /** Stores the companion object */
  def property: DatasetProperty

  /** Applies a sequence of pre-processing functions to the given DataFrame */
  def preprocess(): DataFrame

}
