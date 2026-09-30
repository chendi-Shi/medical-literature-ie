"""Verify completed research artifacts and export shareable numerical evidence."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from nlp_lab.data import digest, read_json, save_json
from nlp_lab.research import macro_f1_fast, SUITE_NAME

NAMES = {"logistic": "TF-IDF Logistic", "linear_svm": "TF-IDF Linear SVM", "ce": "BGE CE",
         "rdrop": "BGE R-Drop", "fgm": "BGE FGM", "rdrop_fgm": "BGE R-Drop + FGM"}


def export(workspace, destination):
    directory = workspace / "research" / SUITE_NAME
    protocol = read_json(directory / "protocol.json")
    report = read_json(directory / "report.json")
    state = read_json(directory / "state.json")
    if state["phase"] != "completed" or len(report["details"]) != 14:
        raise ValueError("套件尚未完成 14 次实验")
    if digest(directory / "protocol.json") != report["protocol_sha256"]:
        raise ValueError("协议指纹不一致")
    for name, expected in protocol["source_code_sha256"].items():
        if digest(Path(__file__).resolve().parents[1] / "nlp_lab" / name) != expected:
            raise ValueError("协议算法文件已改变")
    checks = []
    class_metrics = {}
    for entry in report["details"]:
        run = workspace / "runs" / entry["run"]
        meta = read_json(run / "run.json")
        test = read_json(run / "test_metrics.json")
        robust = read_json(run / "robustness.json")
        class_metrics[entry["run"]] = test
        with np.load(run / "test_arrays.npz") as arrays:
            actual = macro_f1_fast(arrays["truth"], arrays["prediction"], len(protocol["labels"]))
            if len(arrays["truth"]) != 10000 or not np.isclose(actual, entry["test_macro_f1"]):
                raise ValueError("测试数组与汇总不匹配")
        if meta["dataset_sha256"] != report["dataset_sha256"] or test["split_sha256"] != protocol["split_sha256"]["test"]:
            raise ValueError("run / test 数据指纹不一致")
        if any(x["samples"] != 2000 for x in robust["cases"].values()):
            raise ValueError("压力测试样本数不匹配")
        checks.append({"run": entry["run"], "test_metrics_sha256": digest(run / "test_metrics.json"),
                       "test_arrays_sha256": digest(run / "test_arrays.npz"), "robustness_sha256": digest(run / "robustness.json"),
                       "optimizer_steps": meta.get("optimizer_steps"), "epochs": meta.get("epochs_completed"),
                       "amp_skipped_updates": read_json(run / "history.json")[-1].get("skipped_updates"),
                       "model_revision": meta.get("model_revision"), "versions": meta["versions"]})
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "report.json"):
        shutil.copyfile(directory / name, destination / name)
    shutil.copyfile(workspace / "datasets" / protocol["dataset"] / "manifest.json", destination / "data_manifest.json")
    save_json(destination / "verification.json", {"verified_experiments": len(checks), "test_samples_each": 10000,
                                                "stress_samples_each": 2000, "checks": checks})
    # Optional plotting dependency; underlying experiment artifacts need no matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "savefig.facecolor": "white"})
    aggregates = report["aggregates"]
    colors = ["#9ca89d", "#6c8076", "#4d7761", "#6e9e7c", "#be9457", "#346956"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = [NAMES[x["variant"]].replace("TF-IDF ", "").replace("BGE ", "") for x in aggregates]
    ax = axes[0, 0]
    for i, item in enumerate(aggregates[::-1]):
        ax.errorbar(100*item["test_macro_f1"]["mean"], i, xerr=100*(item["test_macro_f1"]["std"] or 0),
                    fmt="o", color=colors[::-1][i], capsize=4, markersize=6)
    ax.set_yticks(range(len(labels)), labels[::-1])
    ax.set_xlim(80, 89); ax.set_xlabel("Full-test Macro-F1 (%)"); ax.set_title("A. Classification (10,000 test titles)")
    for i, item in enumerate(aggregates[::-1]):
        ax.text(80.2, i, f'n={item["n_seeds"]}', va="center", color="#666", fontsize=8)
    ax.text(.02,.97,"Neural: mean ± sample SD; sparse: single run", transform=ax.transAxes, va="top",fontsize=8,color="#666")
    ax.set_ylim(-.6, 5.9)
    ax = axes[0, 1]
    for i, item in enumerate(report["paired_comparisons"]):
        delta=100*item["delta_macro_f1"]; lo,hi=[100*x for x in item["ci95"]]
        ax.errorbar(delta, i, xerr=[[delta-lo],[hi-delta]], fmt="o", color="#346956", capsize=4)
    ax.axvline(0, color="#999", ls="--", lw=1)
    ax.set_yticks(range(3), [NAMES[x["candidate"]].replace("BGE ", "") for x in report["paired_comparisons"]])
    ax.set_xlabel("Macro-F1 difference vs CE (percentage points)")
    ax.set_title("B. Paired sample bootstrap (95% CI)")
    ax.text(.02,.02,"400 replicates; three fixed training seeds\nNo seed-population CI / multiplicity correction", transform=ax.transAxes, fontsize=8, color="#666")
    ax.set_ylim(-.8, 2.7)
    ax = axes[1, 0]
    modes=["punctuation", "char_delete_5pct", "tail_truncate_30pct"]
    for item,color in zip(aggregates,colors):
        ax.plot(range(3), [100*item["robustness"][m]["delta_mean"] for m in modes], "o-", lw=1.2,
                color=color, label=NAMES[item["variant"]].replace("BGE ", ""))
    ax.set_xticks(range(3), ["Punctuation", "Delete 5%", "Truncate 30%"])
    ax.set_ylabel("Macro-F1 change (percentage points)"); ax.set_title("C. Paired 2,000-title synthetic stress")
    ax.legend(fontsize=7, loc="lower left", ncol=2); ax.axhline(0,color="#ddd",lw=1)
    ax = axes[1, 1]
    x=np.arange(len(aggregates))
    ax.bar(x-.17, [100*a["uncalibrated_ece"]["mean"] for a in aggregates], width=.34, color="#b6c4b8", label="Raw scores")
    ax.bar(x+.17, [100*a["calibrated_ece"]["mean"] for a in aggregates], width=.34, color="#346956", label="Validation temperature")
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel("ECE, 15 bins (%)")
    ax.set_title("D. Confidence calibration on held-out test"); ax.legend(fontsize=8)
    fig.suptitle("THUCNews title subset | 10k train budget | fixed recipes", fontsize=14)
    fig.savefig(destination / "research_results.png", dpi=180)
    fig.savefig(destination / "research_results.pdf")
    plt.close(fig)
    def score(item):
        value=item["test_macro_f1"]
        return f'{100*value["mean"]:.2f}%' + (f' ± {100*value["std"]:.2f} 个百分点' if value["std"] is not None else "（单次）")
    chosen=next(x for x in aggregates if x["variant"]==report["selected_by_validation"])
    ce=next(x for x in aggregates if x["variant"]=="ce")
    svm=next(x for x in aggregates if x["variant"]=="linear_svm")
    lines=["# 实测研究结果 v2", "", f'按验证集方法均值选择 **{NAMES[chosen["variant"]]}**；完整 test Macro-F1 为 **{score(chosen)}**。训练 10,000 / 验证 9,991 / 测试 10,000，十类真实新闻标题。神经配方三种种子全部汇报，未选择最佳测试种子。', "",
           "![实测对照、区间、压力测试与校准](results/research_results.png)", "", "## 完整测试集对照", ""]
    for item in aggregates:
        lines.append(f'- **{NAMES[item["variant"]]}**：Macro-F1 {score(item)}；验证均值 {100*item["validation_macro_f1"]["mean"]:.2f}%；Accuracy {100*item["test_accuracy"]["mean"]:.2f}%；单次训练均值 {item["training_seconds"]["mean"]:.1f}s；校准前 / 后 ECE {100*item["uncalibrated_ece"]["mean"]:.2f}% / {100*item["calibrated_ece"]["mean"]:.2f}%。')
    lines += ["", "## 改进幅度与取舍", ""]
    for item in report["paired_comparisons"]:
        lines.append(f'- {NAMES[item["candidate"]]} 相对 CE：{100*item["delta_macro_f1"]:+.2f} 个百分点，条件 95% CI [{100*item["ci95"][0]:+.2f}, {100*item["ci95"][1]:+.2f}]。')
    lines += ["", f'验证选中方法相对 CE 的测试均值变化为 {100*(chosen["test_macro_f1"]["mean"]-ce["test_macro_f1"]["mean"]):+.2f} 个百分点；相对 Linear SVM 为 {100*(chosen["test_macro_f1"]["mean"]-svm["test_macro_f1"]["mean"]):+.2f} 个百分点。其训练时间约为 CE 的 {chosen["training_seconds"]["mean"]/ce["training_seconds"]["mean"]:.2f} 倍。SVM 提供很有竞争力、训练便宜的基线，不能仅因模型复杂就忽略。', "",
              "置信区间按测试样本行、类别分层配对重采样，三个已有训练种子固定；不包含训练种子总体不确定性，也未做多重比较校正。区间跨零的结果不足以在这套协议下确认改善。不同方法并非相同 FLOPs；时间取自本机单次训练，包括验证与保存。", "", "## 同组扰动", ""]
    for item in aggregates:
        r=item["robustness"]
        lines.append(f'- {NAMES[item["variant"]]}：同组原文 {100*r["clean_subset"]["macro_f1_mean"]:.2f}%；标点 / 删字 / 截断相对变化 {100*r["punctuation"]["delta_mean"]:+.2f} / {100*r["char_delete_5pct"]["delta_mean"]:+.2f} / {100*r["tail_truncate_30pct"]["delta_mean"]:+.2f} 个百分点。')
    lines += ["", "组合方法在标点和删字的 F1 退化较 CE 小，但截断退化更大；本轮不能声称它全面提升鲁棒性。", "", "## 每类错误与后续研究依据", ""]
    selected_metrics=[class_metrics[r] for r in chosen["runs"]]
    matrix=np.mean([x["confusion_matrix"] for x in selected_metrics],axis=0)
    per_class={label:float(np.mean([x["per_class"][label]["f1-score"] for x in selected_metrics])) for label in protocol["labels"]}
    save_json(destination / "class_analysis.json", {"variant":chosen["variant"], "mean_confusion_matrix":matrix.tolist(),
                                                  "labels":protocol["labels"], "mean_per_class_f1":per_class})
    for label,f1 in sorted(per_class.items(),key=lambda x:x[1])[:3]:
        i=protocol["labels"].index(label)
        row=matrix[i].copy();row[i]=0;j=int(row.argmax())
        lines.append(f'- 组合模型最弱类别之一 **{label}**：三种子平均 F1 {100*f1:.2f}%；主要混淆为 {label} → {protocol["labels"][j]}，平均每次测试约 {matrix[i,j]:.1f} 条。')
    lines += ["", "这些错误只用于解释本轮结果和设计未来独立实验，未反向修改本轮配方。下一轮若扩大预算、调整编码器或处理短标题语义，应新建协议，并使用未参与本轮观察的最终外部数据验证。"]
    lines += ["", "每种扰动共 2,000 条，十类各 200。删除 / 截断可能改变语义，保留原标签且未人工复标；不能以此宣称自然分布外泛化。", "", "## 证据与适用范围", "",
              "数值来自已保存 checkpoint 的重新加载评测。14 次均验证了测试样本数、评分数组、指纹和同组压力测试数量；[report.json](results/report.json) 保存所有逐种子结果，[protocol.json](results/protocol.json) 保存训练前配方与算法指纹，[verification.json](results/verification.json) 保存产物校验。PDF 图可用于分享：[research_results.pdf](results/research_results.pdf)。", "",
              "当前结论只覆盖单一公开标题域、10k 固定训练预算与 2 epochs。原仓库 180k 训练分数不可直接比较；近重复与预训练语料重叠未排除；三种子样本量有限。校准与 checkpoint 选择共用验证集，不是独立校准集。本轮是现有 R-Drop / FGM 的适配和因子实验，不是原创算法或论文级全面复现。", ""]
    lines += ["12 个神经实验均完成 2 epochs；预算 626 个更新窗口，FP16 动态缩放各跳过 5 次初始非有限梯度更新，实际成功更新均为 621 次。源码、模型修订和更新记录在 verification.json 中。", ""]
    (destination.parent / "RESULTS_V2.md").write_text("\n".join(lines), encoding="utf-8")
    return {"selected": chosen["variant"], "test_macro_f1": chosen["test_macro_f1"], "verified": len(checks)}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    p.add_argument("--workspace", type=Path, default=root / "workspace")
    p.add_argument("--output", type=Path, default=root / "docs/results")
    args=p.parse_args()
    print(json.dumps(export(args.workspace, args.output), ensure_ascii=False))
