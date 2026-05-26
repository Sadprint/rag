"""
RAG 样本收集 — 执行 RAG 链，生成 (问题, 回答, 检索文档, 标注) 四元组
保存为 JSON 中间文件，供 evaluate_metrics.py 反复评估

用法:
  python evaluate_generation.py                         交互模式
  python evaluate_generation.py --max-items 200         收集 200 条
  python evaluate_generation.py --output samples.json   指定输出文件
"""
import argparse
import json
import os
import sys
import time
import types

# ── onnxruntime mock（同 app.py）──
class _FakeORT(types.ModuleType):
    def __init__(self, name): super().__init__(name)
    @staticmethod
    def get_available_providers(): return ["CPUExecutionProvider"]
    @staticmethod
    def InferenceSession(*a, **kw): return types.SimpleNamespace()

def _register(name):
    import importlib.machinery as _machinery
    mod = _FakeORT(name) if name == "onnxruntime" else types.ModuleType(name)
    mod.__spec__ = _machinery.ModuleSpec(name, loader=None, origin="mock")
    mod.__package__ = name.rpartition(".")[0] or ""
    sys.modules[name] = mod

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
warnings.filterwarnings("ignore")

import config
from rag_chain import (
    load_embedding_model,
    load_t2ranking_queries,
    load_t2ranking_passages,
    load_t2ranking_vectorstore,
    run_rag_query,
)


def _resolve_device(args_device):
    import torch
    if args_device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU\n")
        return "cpu"
    return args_device


def _build_pid_lookup(passages):
    """pid → text 查找表，用于构建 reference"""
    return {p.metadata["pid"]: p.page_content for p in passages}


def collect_samples(vs, queries, pid_lookup=None, search_type="mmr", k=None, verbose=True):
    """对每条 query 执行 RAG 链，返回样本列表（纯 dict，不依赖 ragas）"""
    samples = []
    skipped = 0
    total = len(queries)
    start = time.time()

    for idx, q in enumerate(queries):
        question = q["query"]
        try:
            answer, contexts = run_rag_query(vs, question, search_type=search_type, k=k)
        except Exception as e:
            skipped += 1
            err = str(e)[:80]
            if "1301" in str(e) or "contentFilter" in str(e):
                err = "内容审核拦截"
            elif "timeout" in str(e).lower():
                err = "超时"
            if verbose:
                print(f"\n  [{idx + 1}/{total}] 跳过：{err}")
            continue

        # 构建 reference（来自标注的相关文档，用于 ContextPrecision）
        reference = None
        if pid_lookup and "relevant_pids" in q:
            ref_texts = []
            for pid in q["relevant_pids"]:
                text = pid_lookup.get(pid)
                if text:
                    ref_texts.append(text)
            if ref_texts:
                reference = " ".join(ref_texts)

        samples.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
        })

        if verbose and ((idx + 1) % max(1, total // 10) == 0):
            elapsed = time.time() - start
            qps = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / qps if qps > 0 else 0
            print(f"  [{idx + 1}/{total}] {elapsed:.0f}s  {qps:.1f} q/s  ETA {eta:.0f}s",
                  end="", flush=True)

    elapsed = time.time() - start
    if verbose:
        failed = total - len(samples)
        print(f"\n  样本收集完成，{len(samples)}/{total} 条，跳过 {failed} 条，耗时 {elapsed:.1f}s")
    return samples, skipped


def save_samples(filepath, samples, meta=None):
    """保存样本到 JSON"""
    data = {
        "meta": meta or {},
        "samples": samples,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"样本已保存到 {filepath} ({len(samples)} 条)")


def _input_int(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  请输入一个整数")


def main():
    parser = argparse.ArgumentParser(description="RAG 样本收集 (T2Ranking queries)")
    parser.add_argument("--max-items", type=int, default=None, help="限制评估条数")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="嵌入设备 (默认 auto)")
    parser.add_argument("--k", type=int, default=config.DEFAULT_K, help=f"检索 top-k (默认 {config.DEFAULT_K})")
    parser.add_argument("--output", type=str, default="results/eval_samples.json",
                        help="样本输出文件 (默认 results/eval_samples.json)")
    parser.add_argument("--eval", action="store_true",
                        help="收集完成后自动调用 evaluate_metrics.py 评估")
    parser.add_argument("--workers", type=int, default=16, help="RAGAS 并发数 (默认 16)")
    parser.add_argument("--eval-output", type=str, default="results/eval_result.csv",
                        help="--eval 模式下的结果输出 (默认 results/eval_result.csv)")
    args = parser.parse_args()

    interactive = len(sys.argv) == 1

    # ── 设备 ──
    device = _resolve_device(args.device)
    import torch
    device_name = f"{device.upper()}" + \
        (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" and torch.cuda.is_available() else "")
    print(f"设备: {device_name}")

    # ── 加载数据 ──
    print("加载数据集...")
    queries = load_t2ranking_queries()
    if not queries:
        print("docs/t2ranking/eval_queries.json 不存在或为空")
        return
    print(f"  queries: {len(queries):,} 条")

    # ── 加载 passages（用于构建 reference）──
    passages = load_t2ranking_passages()
    pid_lookup = _build_pid_lookup(passages) if passages else {}
    if pid_lookup:
        print(f"  passages: {len(pid_lookup):,} 条")

    # ── 加载嵌入模型 ──
    print("加载嵌入模型...")
    config.EMBEDDING_DEVICE = device
    embeddings = load_embedding_model()
    print("  完成")

    # ── 加载向量库 ──
    print("加载 T2Ranking 向量库...")
    vs = load_t2ranking_vectorstore(embeddings)
    if vs is None:
        print("T2Ranking 向量库不存在，请先运行 evaluate.py --build")
        return
    print(f"  完成 ({vs._collection.count():,} 条)")

    # ── 确定条数 ──
    num_items = args.max_items
    if interactive:
        print()
        num_items = _input_int(
            f"评估多少条查询？（回车=建议 20 条，全量={len(queries):,}）: ",
            default=20,
        )
        print()

    eval_queries = queries if num_items is None else queries[:num_items]

    # ── 收集样本 ──
    print(f"收集 RAG 样本（{len(eval_queries)} 条查询，k={args.k}）...")
    samples, skipped = collect_samples(vs, eval_queries, pid_lookup=pid_lookup,
                                       search_type="mmr", k=args.k)

    if not samples:
        print("没有成功生成任何样本")
        return

    # ── 保存 ──
    meta = {
        "k": args.k,
        "search_type": "mmr",
        "total_queries": len(eval_queries),
        "collected": len(samples),
        "skipped": skipped,
    }
    save_samples(args.output, samples, meta)

    if args.eval:
        print(f"\n── 自动评估 ──")
        import subprocess
        subprocess.run([
            sys.executable, "evaluate_metrics.py",
            "--input", args.output,
            "--output", args.eval_output,
            "--workers", str(args.workers),
        ])
    else:
        print(f"\n下一步: python evaluate_metrics.py --input {args.output}")


if __name__ == "__main__":
    main()
