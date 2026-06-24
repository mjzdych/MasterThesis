import os, re, pandas as pd

BASE_OUT = "/gpfs/home2/mzdych/thesis/experiments_transformer_3d"

regions = [
    # "full_europe_2003", "full_europe_2010", "full_europe_2018",
    # "north_europe_2010", "north_europe_2018", "south_europe_2003",
    "iberia_2003", "mediterranean_2003", "eastern_europe_2010", "scandinavia_2018"
]
ablations = ["cn_era5"]
tasks     = ["bc", "dc", "cc"]

rows = []
for region in regions:
    for task in tasks:
        for abl in ablations:
            run_dir = os.path.join(BASE_OUT, f"{region}_{task}_t3_{abl}_transformer")
            metrics_path = os.path.join(run_dir, "metrics.txt")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as f:
                txt = f.read()
            test_block = re.split(r"=== TEST ===", txt)[-1]
            row = {"region": region, "task": task.upper(), "ablation": abl}
            if task in ("cc", "bc", "dc"):
                for key in ["r2", "pearson", "mae"]:
                    m = re.search(rf"{key}:\s*([-0-9.]+)", test_block)
                    row[key] = float(m.group(1)) if m else None
            elif task == "hw":
                for key in ["roc", "pr_auc", "best_f1", "iou"]:
                    m = re.search(rf"{key}:\s*([-0-9.]+)", test_block)
                    row[key] = float(m.group(1)) if m else None
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("/gpfs/home2/mzdych/thesis/all_results_summary_bc_dc_3d_transformer.csv", index=False)
print(df.to_string(index=False))