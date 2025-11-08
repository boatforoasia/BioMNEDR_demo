import numpy as np
import pandas as pd
import xgboost as xgb
import os
import ast

FEATURE_LIST = [
    "feature/d2p2i_1000_100_++_s128_w5_n5_ts1e9.txt",
    "feature/d2p2p2i_1000_100_++_s128_w5_n5_ts1e9.txt",
    "feature/d2p2f2p2i_1000_100_++_s128_w5_n5_ts1e9.txt",
    "feature/d2p2f2f2p2i_1000_100_++_s128_w5_n5_ts1e9.txt",
]
MODEL_LIST = [
    "model/BioMNEDR_feat_0.model",
    "model/BioMNEDR_feat_1.model",
    "model/BioMNEDR_feat_2.model",
    "model/BioMNEDR_feat_3.model",
]
INDICATION_ID = "C0030567"  # Parkinson's Disease
ASSOCIATION_PATH = "data/6_drug_indication_df.tsv"
IDX2NODE_PATH = "data/idx2node.txt"

def load_idx2node(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    idx2node = ast.literal_eval(content)
    return {int(k): str(v) for k, v in idx2node.items()}

def load_features(feature_file_path, idx2node):
    features = {}
    with open(feature_file_path, "r", encoding="utf-8") as file:
        next(file); next(file)
        for line in file:
            parts = line.strip().split()
            node_idx = int(parts[0][2:])
            node = idx2node[node_idx]
            features[node] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return features

def main():
    idx2node = load_idx2node(IDX2NODE_PATH)

    df = pd.read_csv(ASSOCIATION_PATH, sep="\t", dtype=str)
    drug_list = df["drug"].unique()

    # load feature maps and check indication vectors
    feature_maps = [load_features(p, idx2node) for p in FEATURE_LIST]
    for i, fmap in enumerate(feature_maps):
        if INDICATION_ID not in fmap:
            raise ValueError(f"Indication {INDICATION_ID} missing in feature file {FEATURE_LIST[i]}")

    # load boosters
    boosters = []
    for mp in MODEL_LIST:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Model file not found: {mp}. Run training script first.")
        b = xgb.Booster()
        b.load_model(mp)
        boosters.append(b)

    # collect drugs present in all feature maps
    retained_drugs = []
    per_model_preds = [[] for _ in boosters]

    for drug in drug_list:
        drug_feats = [fmap.get(drug) for fmap in feature_maps]
        if any(f is None for f in drug_feats):
            continue
        retained_drugs.append(drug)
        for i, b in enumerate(boosters):
            X = np.concatenate([drug_feats[i], feature_maps[i][INDICATION_ID]]).reshape(1, -1)
            dtest = xgb.DMatrix(X)
            per_model_preds[i].append(b.predict(dtest)[0])

    if not retained_drugs:
        raise ValueError("No drugs have complete feature vectors across all feature files.")

    preds_array = np.vstack([np.array(p) for p in per_model_preds])  # shape (4, N)
    final_preds = np.max(preds_array, axis=0)

    results = (
        pd.DataFrame({"Drug": retained_drugs, "Prediction": final_preds})
        .sort_values("Prediction", ascending=False)
        .reset_index(drop=True)
    )
    pd.set_option("display.max_rows", None)
    print(results)

if __name__ == "__main__":
    main()
