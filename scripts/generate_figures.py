import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from brats_jepa.config import FIGURES_DIR, METRICS_DIR, ensure_directories


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Figures from Benchmark Summaries")
    parser.add_argument("--metrics_dir", type=str, default=None, help="Directory containing summary CSV files")
    parser.add_argument("--figures_dir", type=str, default=None, help="Directory to save generated figures")
    return parser.parse_args()


def load_json_metrics(json_path):
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    return None

def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir).resolve() if args.metrics_dir else METRICS_DIR
    figures_dir = Path(args.figures_dir).resolve() if args.figures_dir else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    def resolve_summary_csv(filename: str) -> Path | None:
        p = metrics_dir / filename
        if p.exists():
            return p
        exp_dir = metrics_dir.parent / "experiments"
        if exp_dir.exists():
            matches = list(exp_dir.glob(f"**/{filename}"))
            if matches:
                matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return matches[0]
        all_matches = list(metrics_dir.parent.glob(f"**/{filename}"))
        if all_matches:
            all_matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return all_matches[0]
        return None

    # 1. Plot Downstream Segmentation Performance Summary (Test Dice & HD95)
    summary_file = resolve_summary_csv("evaluation_benchmark_summary.csv")
    if summary_file and summary_file.exists():
        df = pd.read_csv(summary_file)
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        df["test_dice_num"] = pd.to_numeric(df["test_dice"], errors="coerce")
        df["hd95_num"] = pd.to_numeric(df["hd95_px"], errors="coerce")
        
        sns.barplot(data=df, x="model", y="test_dice_num", hue="model", legend=False, ax=ax1, palette="viridis")
        ax1.set_title("Downstream Test Dice Similarity (Higher = Better)")
        ax1.set_ylabel("Test Dice Score")
        ax1.tick_params(axis="x", rotation=15)
        
        sns.barplot(data=df, x="model", y="hd95_num", hue="model", legend=False, ax=ax2, palette="rocket")
        ax2.set_title("95th Percentile Hausdorff Distance (Lower = Better)")
        ax2.set_ylabel("HD95 Distance (Pixels)")
        ax2.tick_params(axis="x", rotation=15)
        
        plt.tight_layout()
        out_fig1 = figures_dir / "segmentation_performance_benchmark.png"
        plt.savefig(out_fig1, dpi=300)
        plt.close()
        print(f"Generated figure: {out_fig1}")

    # 2. Representation Collapse Benchmark Plot
    if summary_file and summary_file.exists():
        df = pd.read_csv(summary_file)
        if "effective_rank" in df.columns and "avg_cosine_sim" in df.columns:
            ssl_df = df[df["effective_rank"] != "N/A (CNN)"].copy()
            if len(ssl_df) > 0:
                ssl_df["effective_rank"] = pd.to_numeric(ssl_df["effective_rank"], errors="coerce")
                ssl_df["avg_cosine_sim"] = pd.to_numeric(ssl_df["avg_cosine_sim"], errors="coerce")
            
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            
            sns.barplot(data=ssl_df, x="model", y="effective_rank", hue="model", legend=False, ax=ax1, palette="crest")
            ax1.set_title("Effective Representation Rank (Higher = Richer)")
            ax1.set_ylabel("Effective Rank")
            ax1.tick_params(axis="x", rotation=15)
            
            sns.barplot(data=ssl_df, x="model", y="avg_cosine_sim", hue="model", legend=False, ax=ax2, palette="magma")
            ax2.set_title("Average Cosine Similarity (Lower = Anti-collapse)")
            ax2.set_ylabel("Pairwise Cosine Similarity")
            ax2.tick_params(axis="x", rotation=15)
            
            plt.tight_layout()
            out_fig2 = figures_dir / "representation_collapse_benchmark.png"
            plt.savefig(out_fig2, dpi=300)
            plt.close()
            print(f"Generated figure: {out_fig2}")

    # 3. Low-Data Label Efficiency Curve Plot (1% to 100% Labels)
    low_data_csv = resolve_summary_csv("low_data_benchmark_summary.csv")
    if low_data_csv and low_data_csv.exists():
        ld_df = pd.read_csv(low_data_csv)
        plt.figure(figsize=(10, 5.5))
        sns.lineplot(data=ld_df, x="label_fraction", y="test_dice", hue="model", style="model", markers=True, dashes=False, linewidth=2.5, markersize=9)
        plt.title("Low-Data Label Efficiency Benchmark (1% to 100% Annotations)")
        plt.xlabel("Percentage of Labeled Training Slices")
        plt.ylabel("Downstream Test Dice Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out_fig_ld = figures_dir / "low_data_label_efficiency.png"
        plt.savefig(out_fig_ld, dpi=300)
        plt.close()
        print(f"Generated figure: {out_fig_ld}")

    # 4. Out-of-Distribution (OOD) Scanner Generalization Plot
    ood_csv = resolve_summary_csv("ood_benchmark_summary.csv")
    if ood_csv and ood_csv.exists():
        ood_df = pd.read_csv(ood_csv)
        plt.figure(figsize=(11, 5.5))
        sns.barplot(data=ood_df, x="domain_shift", y="test_dice", hue="model", palette="Set2")
        plt.title("Out-of-Distribution (OOD) Scanner Domain Generalization")
        plt.xlabel("Scanner Domain Shift Condition")
        plt.ylabel("Test Dice Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out_fig_ood = figures_dir / "ood_domain_generalization.png"
        plt.savefig(out_fig_ood, dpi=300)
        plt.close()
        print(f"Generated figure: {out_fig_ood}")

    # 5. BraTS-MEN-RT Cross-Pathology & Missing-Modality OOD Plot
    men_ood_csv = resolve_summary_csv("men_rt_ood_benchmark_summary.csv")
    if men_ood_csv and men_ood_csv.exists():
        men_df = pd.read_csv(men_ood_csv)
        plt.figure(figsize=(11, 5.5))
        sns.barplot(data=men_df, x="adaptation_strategy", y="men_rt_test_dice", hue="model", palette="Accent")
        plt.title("BraTS-MEN-RT Zero-Shot Cross-Pathology & Missing-Modality OOD")
        plt.xlabel("4-Channel Adaptation Strategy")
        plt.ylabel("Meningioma Test Dice Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out_fig_men = figures_dir / "men_rt_ood_generalization.png"
        plt.savefig(out_fig_men, dpi=300)
        plt.close()
        print(f"Generated figure: {out_fig_men}")

if __name__ == "__main__":
    main()
