# if we want to activate the python environment
source .venv/bin/activate

--------------------------------------------------
python - <<'PY'                                                                                          
import json
from pathlib import Path

run = Path("models/llava_runs/llava_batch_test")

for f in list((run/"raw_outputs").glob("*.json"))[:20]:
    d = json.loads(f.read_text())

    if d["true"] != "no_damage":
        print("="*100)
        print("TRUE:", d["true"])
        print("PRED:", d["pred"])
        print("RAW:")
        print(d["raw_llava"][:1000])
        break
PY
-------------------------------------------

how to run llava pipeline? 

# you need to install these packages first 
pip install pydantic

# to run all properties
for p in 001 002 003 004 005 006 007 008; do
  python run_property.py $p
done

in this expiriment i used gpu h100 from snellius with 1 gpu and 18 cpu 1 node
