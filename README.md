# Production Line Performance

Compilation:
```bash
sbt clean package
```

Local execution:
```bash
spark-submit --class "PerformanceEvaluator" --master "local[*]" "target/scala-2.12/production-line-performance_2.12-1.0.jar" --input-path "datasets/adult.data" --classifier-name "DT"
```
