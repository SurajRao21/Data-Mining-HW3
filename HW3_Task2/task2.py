import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate

warnings.filterwarnings("ignore")
np.random.seed(0)

DATA_FILE = "ratings_small.csv"
OUT_CSV = "task2_results.csv"
OUT_SIM_PLOT = "similarity_results.png"
OUT_K_PLOT = "k_results.png"

N_FOLDS = 5
RANDOM_STATE = 0

SIMILARITIES = ["cosine", "msd", "pearson"]

K_VALUES = [5, 10, 20, 30, 50]

MIN_K = 1

SVD_PARAMS = {"n_factors": 50, "random_state": RANDOM_STATE}

CV_N_JOBS = 1


def prepare_noheader_file(src_file, dest_file="ratings_noheader.csv"):
    df = pd.read_csv(src_file)
    df.to_csv(dest_file, index=False, header=False)
    return dest_file

def load_surprise_dataset(noheader_file):
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1, 5))
    data = Dataset.load_from_file(noheader_file, reader=reader)
    return data

def evaluate_algo_cv(algo, data, algo_name):
    start = time.time()
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=N_FOLDS, verbose=False, n_jobs=CV_N_JOBS)
    elapsed = time.time() - start
    mean_rmse = np.mean(results['test_rmse'])
    mean_mae = np.mean(results['test_mae'])
    mean_fit = np.mean(results['fit_time'])
    mean_test = np.mean(results['test_time'])
    return {"name": algo_name, "rmse": mean_rmse, "mae": mean_mae, "fit_time": mean_fit, "test_time": mean_test, "elapsed": elapsed}

def run_all():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found in current folder ({os.getcwd()}). Place ratings_small.csv here.")

    print("Preparing data (remove header if present)...")
    noheader = prepare_noheader_file(DATA_FILE, dest_file="ratings_noheader.csv")
    data = load_surprise_dataset(noheader)

    rows = []

    print("\n=== Running (c) PMF (SVD) ===")
    svd = SVD(**SVD_PARAMS)
    res = evaluate_algo_cv(svd, data, "SVD (PMF)")
    print(f"SVD -> RMSE: {res['rmse']:.6f}, MAE: {res['mae']:.6f}, elapsed(s): {res['elapsed']:.1f}")
    rows.append({
        "Algorithm": "SVD (PMF)",
        "Setting": "SVD default",
        "Similarity": "",
        "K": "",
        "Avg_RMSE": res['rmse'],
        "Avg_MAE": res['mae'],
        "Fit_time": res['fit_time'],
        "Test_time": res['test_time'],
        "Elapsed_s": res['elapsed']
    })

    print("\n=== Running KNNBasic baseline (User & Item, default cosine) ===")
    for user_based in [True, False]:
        name = "User-based CF" if user_based else "Item-based CF"
        sim_options = {"name": "cosine", "user_based": user_based}
        algo = KNNBasic(sim_options=sim_options, k=40, min_k=MIN_K)  
        res = evaluate_algo_cv(algo, data, f"KNNBasic ({name}) cosine")
        print(f"{name} (cosine) -> RMSE: {res['rmse']:.6f}, MAE: {res['mae']:.6f}, elapsed(s): {res['elapsed']:.1f}")
        rows.append({
            "Algorithm": "KNNBasic",
            "Setting": name,
            "Similarity": "cosine",
            "K": 40,
            "Avg_RMSE": res['rmse'],
            "Avg_MAE": res['mae'],
            "Fit_time": res['fit_time'],
            "Test_time": res['test_time'],
            "Elapsed_s": res['elapsed']
        })

    print("\n=== (e) Evaluating similarity metrics ===")
    for sim in SIMILARITIES:
        for user_based in [True, False]:
            cf_type = "User-based" if user_based else "Item-based"
            sim_options = {"name": sim, "user_based": user_based}
            algo = KNNBasic(sim_options=sim_options, k=40, min_k=MIN_K)
            res = evaluate_algo_cv(algo, data, f"KNNBasic ({cf_type}) {sim}")
            print(f"{cf_type} - {sim}: RMSE={res['rmse']:.6f}, MAE={res['mae']:.6f}")
            rows.append({
                "Algorithm": "KNNBasic",
                "Setting": cf_type,
                "Similarity": sim,
                "K": 40,
                "Avg_RMSE": res['rmse'],
                "Avg_MAE": res['mae'],
                "Fit_time": res['fit_time'],
                "Test_time": res['test_time'],
                "Elapsed_s": res['elapsed']
            })

    print("\n=== (f) Evaluating effect of number of neighbors (K) ===")
    for k in K_VALUES:
        for user_based in [True, False]:
            cf_type = "User-based" if user_based else "Item-based"
            sim_options = {"name": "cosine", "user_based": user_based}
            algo = KNNBasic(k=k, sim_options=sim_options, min_k=MIN_K)
            res = evaluate_algo_cv(algo, data, f"KNNBasic ({cf_type}) k={k}")
            print(f"{cf_type} k={k}: RMSE={res['rmse']:.6f}, MAE={res['mae']:.6f}")
            rows.append({
                "Algorithm": "KNNBasic",
                "Setting": cf_type,
                "Similarity": "cosine",
                "K": k,
                "Avg_RMSE": res['rmse'],
                "Avg_MAE": res['mae'],
                "Fit_time": res['fit_time'],
                "Test_time": res['test_time'],
                "Elapsed_s": res['elapsed']
            })

    df = pd.DataFrame(rows)
    df = df[["Algorithm", "Setting", "Similarity", "K", "Avg_RMSE", "Avg_MAE", "Fit_time", "Test_time", "Elapsed_s"]]
    df.to_csv(OUT_CSV, index=False)
    print(f"\nAll results saved to {OUT_CSV}")

    try:
        sim_df = df[(df["K"] == 40) & (df["Algorithm"] == "KNNBasic")].copy()
        if not sim_df.empty:
            fig, ax = plt.subplots(figsize=(8,4))
            pivot_rmse = sim_df.pivot(index="Similarity", columns="Setting", values="Avg_RMSE")
            pivot_rmse.plot(kind="bar", ax=ax)
            ax.set_ylabel("Avg RMSE")
            ax.set_title("Effect of similarity metric on RMSE (KNNBasic, K=40)")
            ax.legend(title="CF Type")
            plt.tight_layout()
            plt.savefig(OUT_SIM_PLOT, dpi=150)
            plt.close(fig)
            print(f"Similarity plot saved to {OUT_SIM_PLOT}")
        else:
            print("No similarity data found for plotting.")

        k_df = df[(df["Similarity"] == "cosine") & (df["Algorithm"] == "KNNBasic")].copy()
        if not k_df.empty:
            fig, ax = plt.subplots(figsize=(8,4))
            for setting in k_df["Setting"].unique():
                subset = k_df[k_df["Setting"] == setting].sort_values("K")
                ax.plot(subset["K"].astype(float), subset["Avg_RMSE"], marker='o', label=setting)
            ax.set_xlabel("K (number of neighbors)")
            ax.set_ylabel("Avg RMSE")
            ax.set_title("RMSE vs K (KNNBasic, cosine)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(OUT_K_PLOT, dpi=150)
            plt.close(fig)
            print(f"K-impact plot saved to {OUT_K_PLOT}")
        else:
            print("No K-varying data found for plotting.")
    except Exception as e:
        print("Plotting failed:", e)

    kruns = df[(df["Algorithm"]=="KNNBasic") & (df["Similarity"]=="cosine") & (df["K"].notnull())]
    best_user = kruns[kruns["Setting"]=="User-based"].sort_values("Avg_RMSE").head(1)
    best_item = kruns[kruns["Setting"]=="Item-based"].sort_values("Avg_RMSE").head(1)
    print("\n=== (g) Best K (by lowest RMSE) ===")
    if not best_user.empty:
        print(f"Best K for User-based CF: {int(best_user['K'].values[0])} (RMSE={best_user['Avg_RMSE'].values[0]:.6f})")
    else:
        print("No User-based K results to report.")
    if not best_item.empty:
        print(f"Best K for Item-based CF: {int(best_item['K'].values[0])} (RMSE={best_item['Avg_RMSE'].values[0]:.6f})")
    else:
        print("No Item-based K results to report.")

    print("\nDONE.")

if __name__ == "__main__":
    run_all()