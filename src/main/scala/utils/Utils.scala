package utils

import scala.reflect.ClassTag
import Numeric.Implicits._

import org.apache.log4j.{Logger => L}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, DataFrame, SparkSession}
import org.apache.spark.sql.functions.{
  col,
  monotonically_increasing_id,
  when,
  rand
}
import org.apache.spark.sql.types.{StructType, DoubleType, StructField}
import org.apache.spark.ml.linalg.{Vector, DenseMatrix}

object Logger extends Serializable {

  @transient lazy val log = L.getLogger(getClass.getName)

  def info(msg: String) = log.info(msg)

  def debug(msg: String) = log.debug(msg)

  def warn(msg: String) = log.warn(msg)

  def error(msg: String) = log.error(msg)

  def fatal(msg: String) = log.fatal(msg)

}

object Utils {

  /** Takes the minimum number of elements in the array,
    *  which sum to a given value or more
    */
  def take(l: Array[Double], limit: Double): Array[Double] = {
    var sum: Double = 0
    l.takeWhile { e =>
      sum += e
      sum <= limit
    }
  }

  /** Computes the average over a generic iterable structure */
  def mean[T: Numeric](xs: Iterable[T]): Double = xs.sum.toDouble / xs.size

  /** Computes the variance over a generic iterable structure */
  def variance[T: Numeric](xs: Iterable[T]): Double = {
    val avg = mean(xs)
    xs.map(_.toDouble).map(a => math.pow(a - avg, 2)).sum / xs.size
  }

  /** Computes the standard deviation over a generic iterable structure */
  def stdDev[T: Numeric](xs: Iterable[T]): Double = math.sqrt(variance(xs))

  /** Converts a DataFrame Row to Array[Double] */
  def rowToArrayOfDouble(record: Row): Array[Double] =
    (for {
      i <- (0 until record.size).filter(x => !record.isNullAt(x))
    } yield record.getDouble(i)).toArray

  /** Converts a DataFrame Column to Array[Double] */
  def colToArrayOfDouble(data: DataFrame, column: String): Array[Double] =
    data.select(column).collect.map(_.getDouble(0))

  /** Defines the seed to be used with randomized operations */
  def seed: Int = 42

  /** Converts a Tuple2[T, T] to Array[T] */
  def tuple2ToArray[T: ClassTag](t: Tuple2[T, T]): Array[T] = Array(t._1, t._2)

  /** Keep a value within a specific range */
  def clip(x: Int, min: Int, max: Int) = math.max(min, math.min(max, x))

  /** Merges two DataFrames, even if they have different columns
    * (though, they must have the same number of rows)
    */
  def mergeDataFrames(dataOne: DataFrame, dataTwo: DataFrame): DataFrame = {
    require(
      dataOne.count == dataTwo.count,
      "The two DataFrames must have the same number of rows."
    )
    val df1 = dataOne.withColumn("_tmp_id", monotonically_increasing_id())
    val df2 = dataTwo.withColumn("_tmp_id", monotonically_increasing_id())
    return df1.join(df2, ("_tmp_id")).drop("_tmp_id")
  }

  /** Adds a column with random 0/1 values to the given DataFrame */
  def addRandomCol(data: DataFrame, colName: String): DataFrame = {
    return data.withColumn(colName, when(rand() > 0.5, 1).otherwise(0))
  }

  /** Generates a DataFrame with the specified number of rows and columns,
    * containing random double values
    */
  def randomDataFrame(
      numRows: Int,
      numCols: Int,
      range: Tuple2[Double, Double] = (0, 100)
  ): DataFrame = {
    val columns = 1 to numCols map (i => "col-" + i)
    val schema = StructType(columns.map(StructField(_, DoubleType)))
    val (a, b) = range
    val rows =
      Seq
        .fill(numRows * numCols)(math.random() * (b - a) + a)
        .grouped(numCols)
        .toList
        .map { x => Row(x: _*) }
    val spark = Spark.session
    val rdd = spark.sparkContext.makeRDD(rows)
    return spark.createDataFrame(rdd, schema)
  }

  /** Splits the given DataFrame based on a specific column.
    * Returns a Map[Any, DataFrame], in which each table contains a different
    * value over the split column and that value is used as the map key.
    */
  def splitDataFrame(
      data: DataFrame,
      splitCol: String
  ): Map[Any, DataFrame] = {
    val states = data.select(splitCol).distinct.collect.flatMap(_.toSeq)
    return states
      .map(state => (state -> data.where(col(splitCol) <=> state)))
      .toMap
  }

  /** Returns the number of rows with each distinct value in the given column */
  def distinctValuesCount(data: DataFrame, column: String): DataFrame = {
    return data.groupBy(column).count()
  }

}

object Spark {

  // Defines the application name
  val APP_NAME = "Production line performance"

  /** Retrieves the current SparkSession or it creates one */
  def session: SparkSession =
    SparkSession.builder.appName(APP_NAME).getOrCreate()

  /** Retrieves the current SparkContext */
  def context: SparkContext = session.sparkContext

  /** Stops Spark execution */
  def stop(): Unit = session.stop()

}
