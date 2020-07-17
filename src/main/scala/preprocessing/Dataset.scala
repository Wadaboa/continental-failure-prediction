package preprocessing

import org.apache.spark.sql.{DataFrame}
import org.apache.spark.sql.types.StructType

trait DatasetProperty {

  /** Specifies discrete columns */
  def getDiscreteColumnNames(): Array[String]

  /** Specifies continuos columns */
  def getContinuosColumnNames(): Array[String]

  /** Specifies target columns */
  def getTargetColumnNames(): Array[String]

  /** Get useful column names */
  def getColumnNames(): Array[String]

  /** Returns the dataset schema string */
  def getSchemaString(): String

  /** Returns the dataset schema */
  def getSchema(): StructType = {
    return StructType.fromDDL(getSchemaString())
  }

}

abstract class Dataset(data: DataFrame) {

  /** Store the companion object */
  def property: DatasetProperty

  /** Check the schema and pre-process the dataset */
  if (!data.schema.equals(property.getSchema()))
    throw new IllegalArgumentException(
      "The given DataFrame does not respect the defined schema."
    )
  preprocess()

  /** Applies a sequence of pre-processing functions to the given DataFrame */
  def preprocess(): DataFrame

  /** Data accessor */
  def getData(): DataFrame = data

}
