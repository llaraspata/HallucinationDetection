import os
import re
import json
import pandas as pd
from src.model.HallucinationDetection import HallucinationDetection


METRICS_FILE_NAME = "metrics_"


def read_metrics(metrics_dir):
    metrics = []

    for activation in HallucinationDetection.ACTIVATION_TARGET:
        metrics_path = os.path.join(metrics_dir, activation)
        files = os.listdir(metrics_path)

        for f in files:
            if f.startswith(METRICS_FILE_NAME) and f.endswith(".json"):
                metrics_path = os.path.join(metrics_path, f)
                match = re.search(r'metrics_layer(\d+).json', f)

                if match:
                    layer = int(match.group(1))
                    with open(metrics_path, "r") as file:
                        m = json.load(file)
                        m["layer"] = layer
                        m["activation"] = activation

                        metrics.append(m)
                else:
                    raise ValueError(f"Layer not found in filename: {f}")

    return pd.DataFrame.from_records(metrics)
