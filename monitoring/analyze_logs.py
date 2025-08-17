import argparse, json, os
import pandas as pd
import matplotlib.pyplot as plt

def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def q(series, p):
    try:
        return float(series.quantile(p))
    except Exception:
        return None

def summarize(edge_df, cloud_df):
    out = {"edge": {}, "cloud": {}, "e2e": {}}
    if not edge_df.empty:
        if "infer_ms" in edge_df:
            out["edge"]["infer_ms_p50"] = round(q(edge_df["infer_ms"], 0.50), 1)
            out["edge"]["infer_ms_p95"] = round(q(edge_df["infer_ms"], 0.95), 1)
        if "fps_ema" in edge_df:
            out["edge"]["fps_ema_mean"] = round(edge_df["fps_ema"].mean(), 2)
        if "cpu" in edge_df:
            out["edge"]["cpu_mean"] = round(edge_df["cpu"].mean(), 1)
        if "mem" in edge_df:
            out["edge"]["mem_mean"] = round(edge_df["mem"].mean(), 1)
        if "cloud_ms" in edge_df:
            s = edge_df["cloud_ms"].dropna()
            if len(s):
                out["edge"]["cloud_rtt_ms_p50"] = round(q(s, 0.50), 1)
                out["edge"]["cloud_rtt_ms_p95"] = round(q(s, 0.95), 1)
        if "persons" in edge_df:
            out["edge"]["frames_with_person"] = int((edge_df["persons"] > 0).sum())
        out["edge"]["frames_total"] = int(len(edge_df))

        # E2E approximation from edge logs: edge_infer_ms + cloud_ms
        if "infer_ms" in edge_df and "cloud_ms" in edge_df:
            e2e = (edge_df["infer_ms"].fillna(0) + edge_df["cloud_ms"].fillna(0))
            e2e = e2e[e2e > 0]  # only where both exist
            if len(e2e):
                out["e2e"]["ms_p50"] = round(q(e2e, 0.50), 1)
                out["e2e"]["ms_p95"] = round(q(e2e, 0.95), 1)

    if not cloud_df.empty and "metrics" in cloud_df:
        # unpack metrics dicts
        recv = cloud_df["metrics"].apply(lambda m: m.get("recv_ms") if isinstance(m, dict) else None).dropna()
        infer = cloud_df["metrics"].apply(lambda m: m.get("infer_ms") if isinstance(m, dict) else None).dropna()
        total = cloud_df["metrics"].apply(lambda m: m.get("total_ms") if isinstance(m, dict) else None).dropna()
        if len(recv):
            out["cloud"]["recv_ms_p50"] = round(q(recv, 0.50), 1)
            out["cloud"]["recv_ms_p95"] = round(q(recv, 0.95), 1)
        if len(infer):
            out["cloud"]["infer_ms_p50"] = round(q(infer, 0.50), 1)
            out["cloud"]["infer_ms_p95"] = round(q(infer, 0.95), 1)
        if len(total):
            out["cloud"]["total_ms_p50"] = round(q(total, 0.50), 1)
            out["cloud"]["total_ms_p95"] = round(q(total, 0.95), 1)
        out["cloud"]["frames"] = int(len(cloud_df))
        if "alert" in cloud_df:
            out["cloud"]["alerts_count"] = int(cloud_df["alert"].notna().sum())

    return out

def save_plots(edge_df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    if not edge_df.empty:
        if "fps_ema" in edge_df:
            ax = edge_df["fps_ema"].plot(title="Edge FPS (EMA)")
            ax.set_xlabel("frame idx"); ax.set_ylabel("fps_ema")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "edge_fps_ema.png")); plt.clf()
        if "infer_ms" in edge_df:
            ax = edge_df["infer_ms"].plot(title="Edge Inference (ms)")
            ax.set_xlabel("frame idx"); ax.set_ylabel("infer_ms")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "edge_infer_ms.png")); plt.clf()
        if "cloud_ms" in edge_df:
            ax = edge_df["cloud_ms"].dropna().plot(title="Edge→Cloud Round Trip (ms)")
            ax.set_xlabel("frame idx"); ax.set_ylabel("cloud_ms")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "edge_cloud_rtt_ms.png")); plt.clf()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edge_jsonl")
    ap.add_argument("cloud_jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    edge_rows = load_jsonl(args.edge_jsonl)
    cloud_rows = load_jsonl(args.cloud_jsonl)
    edge_df = pd.DataFrame(edge_rows)
    cloud_df = pd.DataFrame(cloud_rows)

    # write CSVs
    if not edge_df.empty: edge_df.to_csv(os.path.join(args.out, "edge_metrics.csv"), index=False)
    if not cloud_df.empty: cloud_df.to_csv(os.path.join(args.out, "cloud_metrics.csv"), index=False)

    S = summarize(edge_df, cloud_df)

    # json + txt summaries
    import json as _json
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        _json.dump(S, f, indent=2)

    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Edge Summary\n")
        for k,v in S.get("edge",{}).items(): f.write(f"- {k}: {v}\n")
        f.write("\nCloud Summary\n")
        for k,v in S.get("cloud",{}).items(): f.write(f"- {k}: {v}\n")
        f.write("\nEnd-to-End (approx: edge_infer_ms + cloud_ms)\n")
        for k,v in S.get("e2e",{}).items(): f.write(f"- {k}: {v}\n")

    if args.plots:
        save_plots(edge_df, args.out)

if __name__ == "__main__":
    main()
