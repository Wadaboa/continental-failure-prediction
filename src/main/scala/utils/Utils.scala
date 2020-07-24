package utils

import org.apache.log4j.Logger

object ProductionLinePerformanceLogger extends Serializable {
  @transient lazy val log = Logger.getLogger(getClass.getName)
}
