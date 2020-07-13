package preprocessing

import org.apache.spark.sql.{DataFrame}
import org.apache.spark.sql.types.StructType


trait DatasetProperty {

  /** Specifies discrete columns */
  def getDiscreteColumnNames(): Array[String]

  /** Specifies continuos columns */
  def getContinuosColumnNames(): Array[String]

  /** Specifies target columns  */
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

trait ReflectionCompanion {

  import scala.reflect.runtime._

  def companionOf[T](implicit tt: TypeTag[T]) = {
    val rootMirror = universe.runtimeMirror(getClass.getClassLoader)
    var classSymbol = rootMirror.classSymbol(getClass)
    val classMirror = rootMirror.reflectClass(classSymbol)
    val moduleMirror = classMirror.companion.get
    moduleMirror.instance.asInstanceOf[T]
  }

}

abstract class Dataset(data: DataFrame) extends ReflectionCompanion {
  
  def companion = companionOf[Dataset]

  /** Applies a sequence of pre-processing functions to the given DataFrame */
  def preprocess(): DataFrame

  /** Data accessor */
  def getData(): DataFrame = data

}
