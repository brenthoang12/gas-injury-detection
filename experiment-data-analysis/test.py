import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["wall_time"] = pd.to_datetime(df["wall_time"])
    t0 = df["wall_time"].iloc[0]
    df["time_s"] = (df["wall_time"] - t0).dt.total_seconds()
    for sensor in ("etoh", "h2s"):
        df[f"{sensor}_diff"] = df[f"{sensor}_vgas"] - df[f"{sensor}_vref"]
    return df


def print_summary(label: str, df: pd.DataFrame, sensor: str) -> None:
    diff = df[f"{sensor}_diff"]
    print(f"\n{label}  (n={len(df)})")
    print(f"  {sensor}_vref  : mean={df[f'{sensor}_vref'].mean():.4f}  std={df[f'{sensor}_vref'].std():.4f}")
    print(f"  {sensor}_vgas  : mean={df[f'{sensor}_vgas'].mean():.4f}  std={df[f'{sensor}_vgas'].std():.4f}")
    print(f"  vgas-vref  : mean={diff.mean():.4f}  std={diff.std():.4f}  "
          f"min={diff.min():.4f}  max={diff.max():.4f}")
    print(f"  vgas<vref  : {(diff < 0).sum()} / {len(df)} ({100*(diff < 0).mean():.1f}%)")
    print(f"  temp_C     : mean={df['temp_C'].mean():.1f}  "
          f"rh_pct mean={df['rh_pct'].mean():.1f}")


def print_delta(old: pd.DataFrame, new: pd.DataFrame, sensor: str) -> None:
    vgas_drop = old[f"{sensor}_vgas"].mean() - new[f"{sensor}_vgas"].mean()
    vref_drop = old[f"{sensor}_vref"].mean() - new[f"{sensor}_vref"].mean()
    print(f"\n── {sensor.upper()} delta old → new ──────────────────────────")
    print(f"  {sensor}_vgas dropped : {vgas_drop:+.4f} V")
    print(f"  {sensor}_vref dropped : {vref_drop:+.4f} V")
    print(f"  net signal loss   : {vgas_drop - vref_drop:+.4f} V")
    print(f"──────────────────────────────────────────────")


def plot_vgas_overlay(old: pd.DataFrame, new: pd.DataFrame, sensor: str) -> None:
    vgas_col = f"{sensor}_vgas"
    t_max = min(old["time_s"].max(), new["time_s"].max())
    old_c = old[old["time_s"] <= t_max]
    new_c = new[new["time_s"] <= t_max]
    _, ax = plt.subplots(figsize=(13, 4))
    ax.plot(old_c["time_s"], old_c[vgas_col], lw=0.8, color="#378ADD",
            label=f"OLD  mean={old_c[vgas_col].mean():.4f} V")
    ax.plot(new_c["time_s"], new_c[vgas_col], lw=0.8, color="#D85A30",
            label=f"NEW  mean={new_c[vgas_col].mean():.4f} V")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("V")
    ax.set_title(f"{sensor.upper()} — vgas: OLD vs NEW")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_comparison(old: pd.DataFrame, new: pd.DataFrame, sensor: str) -> None:
    diff_col = f"{sensor}_diff"
    vgas_col = f"{sensor}_vgas"
    vref_col = f"{sensor}_vref"

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── Row 0: vgas and vref over time ──
    for ax, df, label, color in [
        (fig.add_subplot(gs[0, 0]), old, "OLD 2026-03-12", "#378ADD"),
        (fig.add_subplot(gs[0, 1]), new, "NEW 2026-03-24", "#D85A30"),
    ]:
        ax.plot(df["time_s"], df[vgas_col], lw=0.8, color=color, label="vgas")
        ax.plot(df["time_s"], df[vref_col], lw=0.8, color="#888780", ls="--", label="vref")
        ax.set_title(f"{label} — vgas vs vref")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("V")
        ax.legend(fontsize=8)

    # ── Row 1: vgas - vref over time ──
    for ax, df, label, color in [
        (fig.add_subplot(gs[1, 0]), old, "OLD 2026-03-12", "#378ADD"),
        (fig.add_subplot(gs[1, 1]), new, "NEW 2026-03-24", "#D85A30"),
    ]:
        ax.plot(df["time_s"], df[diff_col], lw=0.8, color=color)
        ax.axhline(0, color="#888780", lw=1.0, ls="--")
        ax.set_title(f"{label} — vgas - vref")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("ΔV")

    # ── Row 2: distribution of vgas - vref ──
    ax_hist = fig.add_subplot(gs[2, :])
    ax_hist.hist(old[diff_col], bins=40, alpha=0.6, color="#378ADD",
                 label=f"OLD  mean={old[diff_col].mean():.4f} V")
    ax_hist.hist(new[diff_col], bins=40, alpha=0.6, color="#D85A30",
                 label=f"NEW  mean={new[diff_col].mean():.4f} V")
    ax_hist.axvline(0, color="black", lw=1.2, ls="--", label="vgas = vref")
    ax_hist.set_xlabel("vgas - vref (V)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"Distribution of vgas - vref: OLD vs NEW")
    ax_hist.legend(fontsize=9)

    plt.suptitle(f"{sensor.upper()} sensor diagnostic — clean air comparison", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def main():
    path_new = "20260324-experiment/open_air_clean.csv"
    path_old = "testrun-h2s-st/readings_20260312_114657.csv"

    old = load(path_old)
    new = load(path_new)

    for sensor in ("etoh", "h2s"):
        print(f"\n{'='*54}")
        print(f"  {sensor.upper()} SENSOR")
        print(f"{'='*54}")
        print_summary("OLD (2026-03-12)", old, sensor)
        print_summary("NEW (2026-03-24)", new, sensor)
        print_delta(old, new, sensor)
        plot_vgas_overlay(old, new, sensor)
        plot_comparison(old, new, sensor)


if __name__ == "__main__":
    main()
