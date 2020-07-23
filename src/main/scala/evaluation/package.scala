import Numeric.Implicits._

import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.{
  Vector => OldVector,
  Vectors => OldVectors
}

package object evaluation {

  /** Returns machine precision value */
  lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  /** A vector with its norm for fast distance computation */
  class VectorWithNorm(val vector: OldVector, val norm: Double)
      extends Serializable {

    def this(vector: OldVector) = this(vector, OldVectors.norm(vector, 2.0))

    def this(array: Array[Double]) = this(OldVectors.dense(array))

    /** Converts the vector to a dense vector */
    def toDense: VectorWithNorm =
      new VectorWithNorm(OldVectors.dense(vector.toArray), norm)
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
    (for (i <- 2 to record.size) yield record.getInt(i).toDouble).toArray

}
