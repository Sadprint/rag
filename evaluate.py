"""
RAG 检索评估 — 交互式 CLI 脚本
指标：Recall@k / MRR / NDCG
数据集：T2Ranking

用法:
  python evaluate.py                     交互模式（推荐）
  python evaluate.py --build             重建向量库 + 全量评估
  python evaluate.py --passages 5000     限制 passage 数量
  python evaluate.py --max-items 100     限制评估条数
  python evaluate.py --output out.csv    导出 CSV
  python evaluate.py --format json       导出 JSON
"""
import argparse
import csv
import gc
import json
import math
import os
import shutil
import sys
import time
import types as _types
import importlib.machinery as _machinery

# ── onnxruntime DLL 在此 Windows 环境损坏，而 chromadb 导入时会在类定义阶段
#    实例化默认 ONNX 嵌入函数。我们用 BGE 嵌入不需要它，mock 掉即可。 ──
class _FakeORT(_types.ModuleType):
    def __init__(self, name): super().__init__(name)
    @staticmethod
    def get_available_providers(): return ["CPUExecutionProvider"]
    @staticmethod
    def InferenceSession(*a, **kw): return _types.SimpleNamespace()

def _register(name):
    mod = _FakeORT(name) if name == "onnxruntime" else _types.ModuleType(name)
    mod.__spec__ = _machinery.ModuleSpec(name, loader=None, origin="mock")
    mod.__package__ = name.rpartition(".")[0] or ""
    sys.modules[name] = mod

# 静默 chromadb 遥测日志
os.environ["ANONYMIZED_TELEMETRY"] = "false"

_register("onnxruntime")
_register("onnxruntime.capi")
_register("onnxruntime.capi._pybind_state")

# Windows GBK 终端强制 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")
from langchain_community.vectorstores import Chroma

import config
from rag_chain import (
    load_embedding_model,
    load_t2ranking_passages,
    load_t2ranking_queries,
    load_t2ranking_vectorstore,
)


# ==================== Build ====================

def build_vectorstore(embeddings, passages, persist_dir, batch_size=500):
    """分批构建 T2Ranking 向量库。
    调用者应在之前释放已有 Chroma 连接并清理 persist_dir。"""
    from tqdm import tqdm

    os.makedirs(persist_dir, exist_ok=True)

    total = len(passages)
    vs = None
    pbar = tqdm(total=total, unit="passage", desc="  构建向量", ncols=80)
    for i in range(0, total, batch_size):
        batch = passages[i:i + batch_size]
        if vs is None:
            vs = Chroma.from_documents(
                documents=batch, embedding=embeddings,
                persist_directory=persist_dir,
                collection_name="t2ranking_eval",
            )
        else:
            vs.add_documents(batch)

        pbar.update(len(batch))
    pbar.close()
    return vs


# ==================== Evaluate ====================

def run_evaluation(vs, queries, k=10, verbose=False):
    """运行检索评估，返回 (summary, details)"""
    total = len(queries)
    print(f"  评估 {total:,} 条查询 (k={k})...")

    details = []
    recall_sum = mrr_sum = ndcg_sum = 0.0
    start = time.time()
    report_interval = max(1, total // 100)  # 每 1% 报告一次

    for idx, q in enumerate(queries):
        qid = q["qid"]
        query_text = q["query"]
        relevant_pids = set(q["relevant_pids"])

        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        docs = retriever.invoke(query_text)
        retrieved_pids = [doc.metadata.get("pid") for doc in docs]

        # Recall@k
        hit = 1.0 if any(pid in relevant_pids for pid in retrieved_pids) else 0.0
        recall_sum += hit

        # MRR
        mrr = 0.0
        for rank, pid in enumerate(retrieved_pids, 1):
            if pid in relevant_pids:
                mrr = 1.0 / rank
                break
        mrr_sum += mrr

        # NDCG
        dcg = 0.0
        for i, pid in enumerate(retrieved_pids, 1):
            if pid in relevant_pids:
                dcg += 1.0 / math.log2(i + 1)
        ideal_hits = min(len(relevant_pids), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_sum += ndcg

        details.append({
            "qid": qid,
            "query": query_text,
            "retrieved_pids": retrieved_pids,
            "relevant_pids": list(relevant_pids),
            "hit": int(hit),
            "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4),
        })

        # 进度
        if (idx + 1) % report_interval == 0 or idx == total - 1:
            elapsed = time.time() - start
            qps = (idx + 1) / elapsed if elapsed > 0 else 0
            pct = (idx + 1) / total * 100
            eta = (total - idx - 1) / qps if qps > 0 else 0
            print(f"\r  [{idx + 1:,}/{total:,}] {pct:.0f}%  {elapsed:.0f}s  {qps:.1f} q/s  ETA {eta:.0f}s",
                  end="", flush=True)

    elapsed = time.time() - start
    print(f"\n  完成，耗时 {elapsed:.1f}s ({total / elapsed:.1f} q/s)")

    summary = {
        "total": total,
        "k": k,
        "recall_at_k": round(recall_sum / total, 4),
        "mrr": round(mrr_sum / total, 4),
        "ndcg": round(ndcg_sum / total, 4),
    }
    return summary, details


# ==================== Output ====================

def print_results(summary, details, verbose=False):
    """格式化输出评估结果"""
    k = summary["k"]
    hits = sum(1 for d in details if d["hit"] == 1)
    misses = len(details) - hits

    print()
    print("=" * 58)
    print("  T2Ranking 检索评估结果")
    print("=" * 58)
    print(f"  查询数:       {summary['total']:,}")
    print(f"  k:            {k}")
    print(f"  命中 / 未命中: {hits:,} / {misses:,}")
    print(f"  ─" * 20)
    print(f"  Recall@{k}:     {summary['recall_at_k']:.4f}")
    print(f"  MRR:           {summary['mrr']:.4f}")
    print(f"  NDCG@{k}:       {summary['ndcg']:.4f}")
    print("=" * 58)

    if verbose and misses > 0:
        missed = [d for d in details if d["hit"] == 0]
        print()
        print(f"  ─── 未命中查询（前 20 / 共 {len(missed)} 条）───")
        for i, d in enumerate(missed[:20], 1):
            print(f"  {i}. {d['query'][:80]}")
            print(f"     检索: {d['retrieved_pids']}")
            print(f"     相关: {d['relevant_pids']}")


def export_csv(filepath, summary, details):
    """导出 CSV"""
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "qid", "query", "hit", "mrr", "ndcg",
            "retrieved_pids", "relevant_pids",
        ])
        for d in details:
            writer.writerow([
                d["qid"], d["query"], d["hit"], d["mrr"], d["ndcg"],
                "|".join(map(str, d["retrieved_pids"])),
                "|".join(map(str, d["relevant_pids"])),
            ])
    print(f"\n📁 结果已导出到 {filepath}")


def export_json(filepath, summary, details):
    """导出 JSON"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)
    print(f"\n📁 结果已导出到 {filepath}")


# ==================== Interactive Prompts ====================

def _input_int(prompt, default=None):
    """交互式读取整数"""
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  请输入一个整数")


def _input_yn(prompt, default="y"):
    """交互式读取 y/n"""
    suffix = "Y/n" if default == "y" else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if raw == "":
        return default == "y"
    return raw in ("y", "yes")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="T2Ranking RAG 检索评估")
    parser.add_argument("--build", action="store_true", help="重建向量库")
    parser.add_argument("--passages", type=int, default=None, help="限制 passage 数量")
    parser.add_argument("--max-items", type=int, default=None, help="限制评估条数")
    parser.add_argument("--k", type=int, default=10, help="检索 top-k (默认 10)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="嵌入设备 (默认 cuda)")
    parser.add_argument("--output", type=str, default=None, help="导出文件路径 (.csv / .json)")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "json"],
                        help="导出格式 (默认 csv)")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示每条查询详情")
    args = parser.parse_args()

    interactive = len(sys.argv) == 1

    # ==================== 设备 ====================
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，回退到 CPU\n")
        config.EMBEDDING_DEVICE = "cpu"
        device_name = "CPU (CUDA 不可用)"
    else:
        config.EMBEDDING_DEVICE = args.device
        device_name = f"{args.device.upper()}" + \
            (f" ({torch.cuda.get_device_name(0)})" if args.device == "cuda" else "")

    # ==================== 数据 ====================
    print(f"🖥️  设备: {device_name}")
    print("📊 加载数据集...")
    passages = load_t2ranking_passages()
    queries = load_t2ranking_queries()

    if not passages:
        print("❌ docs/t2ranking/passages.json 不存在或为空")
        return
    if not queries:
        print("❌ docs/t2ranking/eval_queries.json 不存在或为空")
        return

    print(f"   passages: {len(passages):,} 条")
    print(f"   queries:  {len(queries):,} 条")

    # ==================== 嵌入模型 ====================
    print("🧠 加载嵌入模型...")
    embeddings = load_embedding_model()
    # 确认模型实际所在设备
    try:
        actual_device = str(embeddings._client.device)
    except Exception:
        actual_device = "未知"
    print(f"   完成 (device={actual_device})")

    # ==================== 向量库 ====================
    persist_dir = os.path.join(str(config.DOCS_DIR), "t2ranking", "chroma_db")
    existing_vs = load_t2ranking_vectorstore(embeddings)
    vs_exists = existing_vs is not None

    if vs_exists:
        count = existing_vs._collection.count()
        print(f"📂 向量库已存在 ({count:,} 条)")
    else:
        print("📂 向量库尚未构建")

    do_build = args.build
    num_passages = args.passages

    if interactive:
        print()
        if vs_exists:
            do_build = _input_yn("是否重建向量库？", default="n")
        else:
            do_build = True

        if do_build:
            num_passages = _input_int(
                f"构建多少条 passage？（回车=全量 {len(passages):,} 条）: ",
                default=len(passages),
            )
        print()

    if do_build:
        n = num_passages or len(passages)
        build_passages = passages if num_passages is None else passages[:num_passages]
        # 通过 existing_vs 的 client 删集合，然后释放它，避免 Windows 文件锁
        if existing_vs is not None:
            try:
                existing_vs._client.delete_collection(existing_vs._collection.name)
            except Exception:
                pass
            del existing_vs
            gc.collect()
            # 等后台线程释放文件句柄
            for _ in range(10):
                try:
                    if os.path.exists(persist_dir):
                        shutil.rmtree(persist_dir)
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(0.5)
        vs = build_vectorstore(embeddings, build_passages, persist_dir)
    else:
        vs = existing_vs
        if vs is None:
            print("❌ 向量库不存在，请先运行 --build 或以交互模式构建")
            return

    # ==================== 评估 ====================
    num_items = args.max_items

    if interactive:
        num_items = _input_int(
            f"评估多少条查询？（回车=全量 {len(queries):,} 条）: ",
            default=len(queries),
        )
        print()

    eval_queries = queries if num_items is None else queries[:num_items]

    try:
        summary, details = run_evaluation(vs, eval_queries, k=args.k, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        return

    print_results(summary, details, verbose=args.verbose)

    # ==================== 导出 ====================
    if args.output:
        if args.output.endswith(".json"):
            export_json(args.output, summary, details)
        else:
            export_csv(args.output, summary, details)
    elif interactive:
        print()
        if _input_yn("导出结果？", default="n"):
            fmt = input("格式 [csv/json] (回车=csv): ").strip().lower() or "csv"
            path = f"eval_result.{fmt}"
            if fmt == "json":
                export_json(path, summary, details)
            else:
                export_csv(path, summary, details)


if __name__ == "__main__":
    main()
