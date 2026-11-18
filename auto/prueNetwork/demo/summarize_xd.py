"""跨域涨点结果汇总: results/<项目>@<域>/seed*.json → 单视/D4/flip4 对比表。"""
import glob
import json
import statistics

MODE_COLS = ("d4", "flip4")


def main():
    print("| 项目@域(涨点域) | ncls | n_test | 单视 macro-F1 | "
          + " | ".join(f"+{m} (Δmacro-F1)" for m in MODE_COLS) + " |")
    print("|---|---:|---:|---:|" + "---:|" * len(MODE_COLS))
    for d in sorted(glob.glob("demo/results/*@*")):
        rs = [json.loads(open(f).read())
              for f in sorted(glob.glob(d + "/seed*.json"))]
        if not rs:
            continue
        name = d.split("/")[-1]
        s = statistics.mean(r["single"]["macro_f1"] for r in rs)
        cells = []
        for m in MODE_COLS:
            if all(m in r for r in rs):
                mv = statistics.mean(r[m]["macro_f1"] for r in rs)
                diffs = [r[m]["macro_f1"] - r["single"]["macro_f1"] for r in rs]
                cells.append(f"{mv:.4f} ({statistics.mean(diffs):+.4f}, "
                             f"{sum(1 for x in diffs if x > 0)}/{len(diffs)})")
            else:
                cells.append("—")
        print(f"| {name} | {rs[0].get('n_classes','?')} | {rs[0]['n_test']} "
              f"| {s:.4f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
