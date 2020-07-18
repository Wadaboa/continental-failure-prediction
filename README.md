# Production Line Performance

Compilation:
```bash
sbt clean package
```

Local execution:
```bash
spark-submit --class "PerformanceEvaluator" --master "local[*]" "target/scala-2.12/production-line-performance_2.12-1.0.jar" --input-path "datasets/adult.data" --classifier-name "DT"
```

Notes:
* AWS Educate Account credentials change after 3 hours (session expired). In order to get them again you have to log in and copy/paste the keys into ~/.aws/credentials (to use `awscli`), or in `aws-config.env` (for this project)
