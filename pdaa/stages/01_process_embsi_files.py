import pandas as pd, pathlib

outdir = pathlib.Path('cache/process_embsi_files')
outdir.mkdir(parents=True, exist_ok=True)

# create parquet files from provided embsi documents
#   - cache/process_embsi_files/dnt.parquet
#   - cache/process_embsi_files/reprotox.parquet

embsi_files = list(f for f in pathlib.Path('resources/embsi').rglob('*.xlsx') if 'processed' in str(f).lower())
dnt_file = next((f for f in embsi_files if 'dnt' in str(f).lower()), None)
reprotox_file = next((f for f in embsi_files if 'reprotox' in str(f).lower()), None)

# process dnt file
lines = pd.read_excel(dnt_file, sheet_name='Endpoint Data Extraction').to_string().split('\n')
header_line = next((i for i, line in enumerate(lines) if 'endpoint' in line), None)
dnt_df = pd.read_excel(dnt_file, sheet_name='Endpoint Data Extraction', skiprows=header_line)
dnt_df = dnt_df.astype(str)
dnt_df.to_parquet(outdir / 'dnt.parquet')

# process reprotox file
lines = pd.read_excel(reprotox_file, sheet_name='Endpoint Data Extraction').to_string().split('\n')
header_line = next((i for i, line in enumerate(lines) if 'Endpoint' in line), None)
reprotox_df = pd.read_excel(reprotox_file, sheet_name='Endpoint Data Extraction', skiprows=header_line)
reprotox_df = reprotox_df.astype(str)
reprotox_df.to_parquet(outdir / 'reprotox.parquet')
