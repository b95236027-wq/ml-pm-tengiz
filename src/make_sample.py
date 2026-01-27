import pandas as pd
import numpy as np

np.random.seed(42)

n = 300  # количество строк sample

data = {
    "sensor1_mean": np.random.normal(0, 1, n),
    "sensor1_std": np.random.uniform(0, 1, n),
    "sensor1_min": np.random.normal(-2, 1, n),
    "sensor1_max": np.random.normal(2, 1, n),

    "sensor2_mean": np.random.normal(5, 2, n),
    "sensor2_std": np.random.uniform(0, 2, n),
    "sensor2_min": np.random.normal(1, 1, n),
    "sensor2_max": np.random.normal(9, 1, n),

    "sensor3_mean": np.random.normal(10, 3, n),
    "sensor3_std": np.random.uniform(0, 3, n),

    # target
    "failure": np.random.binomial(1, 0.2, n)
}

df = pd.DataFrame(data)

df.to_csv("data/sample/features_sample.csv", index=False)

print("Sample dataset created: data/sample/features_sample.csv")
