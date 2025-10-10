import os
import orjson
import evaluation
from pathlib import Path

baseline_name = "random_baseline" # change this for other baselines

workspace = Path("dataset")
test_dir = workspace / "test"
results_dir = workspace / "results" / baseline_name

test_files = sorted(os.listdir(test_dir))
result_files = sorted(os.listdir(results_dir))

test_files = [test_dir / f for f in test_files]
result_files = [results_dir / f for f in result_files]

P3 = 0
P5 = 0
P10 = 0
PR = 0
MRR = 0

for test_file, result_file in zip(test_files, result_files):
    with open(test_file, "rb") as f:
        test_data = orjson.loads(f.read())
    with open(result_file, "rb") as f:
        result_data = orjson.loads(f.read())
    
    measures = evaluation.evaluation_report(result_data, test_data)
    P3 += measures["P@3"]["mean"]
    P5 += measures["P@5"]["mean"]
    P10 += measures["P@10"]["mean"]
    PR += measures["P@R"]["mean"]
    MRR += measures["RR"]["mean"]

n = len(test_files)
print(f"Results for {baseline_name}:")
print(f"P@3: {P3/n:.5f}")
print(f"P@5: {P5/n:.5f}")
print(f"P@10: {P10/n:.5f}")
print(f"P@R: {PR/n:.5f}")
print(f"MRR: {MRR/n:.5f}")