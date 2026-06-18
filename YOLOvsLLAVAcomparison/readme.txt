# if we want to activate the python environment
source .venv/bin/activate

--------------------------------------------------
# if you want to see what llava os producing
python - <<'PY'
import pandas as pd

df = pd.read_csv("models/llava_runs/llava_baseline_results.csv")

for i,row in df.head(15).iterrows():
    print("="*80)
    print("TRUE :", row["true"])
    print("PRED :", row["pred"])
    print("RAW  :", row["raw_llava"])
PY
-------------------------------------------