package evaluation

import org.apache.spark.sql.DataFrame
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
    val mcc = (tp * tn - fp * fn) / math.sqrt(
      (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    return mcc
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
    val tn = cfm(0, 0)
    val fp = cfm(0, 1)
    val fn = cfm(1, 0)
    val tp = cfm(1, 1)
    return (tn, fp, fn, tp)
  }

}
