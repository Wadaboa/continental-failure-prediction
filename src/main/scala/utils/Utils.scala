package utils

import scala.reflect.ClassTag
import Numeric.Implicits._

import org.apache.log4j.{Logger => L}
import org.apache.spark.sql.Row

object Logger extends Serializable {

  @transient lazy val log = L.getLogger(getClass.getName)

  def info(msg: String) = log.info(msg)

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

  def seed: Int = 42

  def tuple2ToArray[T: ClassTag](t: Tuple2[T, T]): Array[T] = Array(t._1, t._2)

}
