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
    Logger.info(s"Confusion matrix: \n${cfm.toString()}")
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

}

object AccuracyByClass {

  def computeAccuracyByClass(
      data: DataFrame,
      predictionCol: String,
      labelCol: String,
      classString: String
  ): Double = {
    val x = data.groupBy(labelCol).count()
    var total: Long = 0
    x.foreach(r => {
      var value = r.get(0).toString()
      var count = r.getAs[Long]("count")
      if (value == classString) total = count
    })

    val y = data
      .filter(col(predictionCol) === col(labelCol))
      .groupBy(predictionCol)
      .count()
    var exact: Long = 0
    y.foreach(r => {
      var value = r.get(0).toString()
      var count = r.getAs[Long]("count")
      if (value == classString) exact = count
    })

    if (total == 0) return 0
    else return exact.asInstanceOf[Double] / total.asInstanceOf[Double]
  }

}
