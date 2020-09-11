package evaluation

import utils._

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object MCC {

  /** Computes Matthew's Correlation Coefficient */
  def computeMccScore(
      data: DataFrame,
      predictionCol: String,
      labelCol: String
  ): Double = {
    val (tn, fp, fn, tp) =
      ConfusionMatrix.computeConfusionMatrix(data, predictionCol, labelCol)
    val num = (tp * tn - fp * fn)
    val den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if (den == 0) {
      return num
    } else {
      return num / den
    }
  }

}

object ConfusionMatrix {

  /** Computes the confusion matrix of a given a DataFrame with labels and predictions */
  def computeConfusionMatrix(
      data: DataFrame,
      predictionCol: String,
      labelCol: String
  ): Tuple4[Double, Double, Double, Double] = {
    val predictionAndLabels = data.select(predictionCol, labelCol)
    val rdd = predictionAndLabels.rdd.map(row => (row.get(0), row.get(1)))
    val metrics = new MulticlassMetrics(rdd)
    val cfm = metrics.confusionMatrix
    if (cfm.numRows == 1 && cfm.numCols == 1) {
      return (0.0, 0.0, 0.0, cfm(0, 0))
    } else {
      val tn = cfm(0, 0)
      val fp = cfm(0, 1)
      val fn = cfm(1, 0)
      val tp = cfm(1, 1)
      return (tn, fp, fn, tp)
    }
  }

  /** Computes the precision score */
  def computePrecision(
      data: DataFrame,
      predictionCol: String,
      labelCol: String
  ): Double = {
    val (_, fp, _, tp) =
      computeConfusionMatrix(data, predictionCol, labelCol)
    val num = tp
    val den = tp + fp
    if (den == 0) return num
    else return num / den
  }

  /** Computes the recall score */
  def computeRecall(
      data: DataFrame,
      predictionCol: String,
      labelCol: String
  ): Double = {
    val (_, _, fn, tp) =
      computeConfusionMatrix(data, predictionCol, labelCol)
    val num = tp
    val den = tp + fn
    if (den == 0) return num
    else return num / den
  }

}
