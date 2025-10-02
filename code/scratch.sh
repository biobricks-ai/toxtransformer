# run 5_0_1_consolidate_evaluations.py
PYTHONPATH=./ python code/5_0_1_consolidate_evaluations.py

# run 5_1_eval_multi_properties.py
PYTHONPATH=./ spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/5_1_eval_multi_properties.py 2> cache/eval_multi_properties/logs/err.log

# run 7_benchmarks.py
PYTHONPATH=./ python code/7_benchmarks.py
