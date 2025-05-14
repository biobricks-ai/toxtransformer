import sqlite3
import pandas as pd
import pathlib
from cvae.models.datasets import PropertyGuaranteeDataset, SharedSampleTracker
from cvae.models.multitask_transformer import SelfiesPropertyValTokenizer

outdir = pathlib.Path('cache/benchmark_evaluations')
outdir.mkdir(parents=True, exist_ok=True)

BENCHMARK_DATASETS = ["BBBP", "Tox21", "CLINTOX", 
                      "BACE", "sider", "chembl", "ctdbase",
                      "ice","toxvaldb","reach"]


# get the data for these benchmark datasets
conn = sqlite3.connect('cache/build_sqlite/cvae.sqlite')
query = lambda q: pd.read_sql_query(q, conn)
res = query('select * from property limit 10')

propdf = query("""
SELECT distinct s.source, p.property_token, pss.positive_count, pss.negative_count
FROM property p 
inner join source s on p.source_id = s.source_id 
inner join property_summary_statistics pss on p.property_id = pss.property_id
where s.source in ('{}')""".format("','".join(BENCHMARK_DATASETS)))

vc1 = propdf['source'].value_counts()

resdf = propdf.copy()
resdf = resdf[resdf['positive_count'] > 20]
resdf = resdf[resdf['negative_count'] > 20]
resdf['weight'] = 1. / resdf.groupby('source')['source'].transform('count')

resdf.to_parquet(outdir / 'benchmark_properties.parquet')