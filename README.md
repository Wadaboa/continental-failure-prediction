# Continental Failure Prediction

Compilation:
```bash
sbt clean package
```

Local execution:
```bash
spark-submit --class "SimpleApp" --master "local[*]" "target/scala-2.12/scala-project_2.12-1.0.jar" --input-path "dataset/adult.data" --classifier-name "DT"
```
