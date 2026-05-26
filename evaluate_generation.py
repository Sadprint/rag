"""
RAG 生成评估 — RAGAS 集成
指标：Faithfulness / Answer Relevancy / Context Utilization
数据集：T2Ranking（复用 queries，无需额外标注）

用法:
  python evaluate_generation.py                       交互模式（推荐）
  python evaluate_generation.py --max-items 50        限制评估条数
  python evaluate_generation.py --output out.csv --format csv
"""
import argparse
import csv
import gc
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

# ── langchain_community.chat_models.vertexai 兼容 mock ──
from langchain_core.language_models.chat_models import BaseChatModel as _BaseChatModel
_mock_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
_mock_vertexai.__package__ = "langchain_community.chat_models"

class _FakeChatVertexAI(_BaseChatModel):
    """placeholder for ragas import compatibility"""
    def _generate(self, *a, **kw): pass
    def _llm_type(self): return "fake"

_mock_vertexai.ChatVertexAI = _FakeChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _mock_vertexai

# Windows GBK 终端强制 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")
warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper.*deprecated.*")
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, ContextUtilization
from ragas.llms import LangchainLLMWrapper

import config
from rag_chain import (
    load_embedding_model,
    load_t2ranking_queries,
    load_t2ranking_vectorstore,
    run_rag_query,
)


def _resolve_device(args_device):
    import torch
    if args_device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU\n")
        return "cpu"
    return args_device


def build_samples(vs, queries, search_type="mmr", k=4, verbose=True):
    """对每条 query 执行 RAG 链，收集 (question, answer, contexts) 样本"""
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

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        ))

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
    return samples


def run_evaluation(samples, max_workers=16, verbose=True):
    """用 RAGAS 并发评估样本集"""
    if not samples:
        return None

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.ZHIPUAI_API_KEY,
        base_url=config.ZHIPUAI_BASE_URL,
        temperature=0.01,
    )
    judge = LangchainLLMWrapper(llm)

    metrics = [
        Faithfulness(llm=judge),
        ContextUtilization(llm=judge),
    ]

    dataset = EvaluationDataset(samples=samples)

    if verbose:
        print(f"\n  正在评估 {len(samples)} 条样本 ({len(metrics)} 个指标)...")
        print(f"  judge: {config.LLM_MODEL} @ {config.ZHIPUAI_BASE_URL}")
        print(f"  workers: {max_workers}")

    from ragas import RunConfig
    run_config = RunConfig(max_workers=max_workers, max_retries=2, timeout=120)

    start = time.time()
    import asyncio
    from ragas.evaluation import aevaluate
    result = asyncio.run(aevaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
    ))
    elapsed = time.time() - start

    if verbose:
        pct = min(100, max_workers)
        speedup = len(samples) * len(metrics) * 12 / max(elapsed, 1)
        print(f"  评估完成，耗时 {elapsed:.1f}s ({len(samples) / elapsed:.1f} 样本/s, {max_workers} 并发)")

    return result, elapsed


def _compute_means(scores_list):
    """scores_list: [{'faithfulness': 0.5, ...}, ...] → {'faithfulness': 0.85, ...}"""
    if not scores_list:
        return {}
    keys = scores_list[0].keys()
    means = {}
    ctx = {}
    for k in keys:
        vals = []
        errors = 0
        import math
        for s in scores_list:
            v = s.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                errors += 1
                continue
            vals.append(v)
        means[k] = sum(vals) / len(vals) if vals else float("nan")
        ctx[f"{k}_errors"] = errors
    means.update(ctx)
    return means


def print_results(result, elapsed, num_samples):
    """格式化输出，含最差样本"""
    print()
    print("=" * 58)
    print("  RAGAS 生成评估结果")
    print("=" * 58)
    print(f"  样本数:        {num_samples}")
    print(f"  评估耗时:      {elapsed:.1f}s")
    print(f"  judge:         {config.LLM_MODEL}")
    print()

    # 均值
    means = _compute_means(result.scores)
    print(f"  {'指标':25s} {'均值':>8s}  {'错误':>5s}")
    print(f"  {'─' * 33}  {'─' * 5}")
    for name in ["faithfulness", "context_utilization"]:
        score = means.get(name, float("nan"))
        errors = means.get(f"{name}_errors", 0)
        print(f"  {name:25s} {score:8.4f}  {errors:5d}")
    print("=" * 58)

    # 列出 faithfulness 最低的 5 条（排除 NaN）
    import math
    valid = [
        (s, d) for s, d in zip(result.scores, result.dataset)
        if isinstance(s.get("faithfulness"), (int, float)) and not math.isnan(s["faithfulness"])
    ]
    if valid:
        scored = sorted(valid, key=lambda x: x[0]["faithfulness"])
        print()
        print(f"  faithfulness 最低的 {min(5, len(scored))} 条（排查幻觉来源）：")
        for i, (scores_dict, sample) in enumerate(scored[:5], 1):
            q = sample.user_input[:60]
            f = scores_dict["faithfulness"]
            print(f"  [{i}] faith={f:.2f}  {q}")
    print()

    print("  指标说明:")
    print("    faithfulness       — 答案中有多少事实可在检索文档中找到（防幻觉）")
    print("    context_utilization — 检索文档中有多少信息被答案所使用")


def export_csv_(filepath, result, samples):
    """导出含逐条分数的 CSV"""
    df = result.to_pandas()
    # 展平 retrieved_contexts 列
    if "retrieved_contexts" in df.columns:
        df["retrieved_contexts"] = df["retrieved_contexts"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else str(x)
        )
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"结果已导出到 {filepath} ({len(df)} 行 × {len(df.columns)} 列)")


def _get_scores(result):
    if hasattr(result, "scores"):
        return _compute_means(result.scores)
    return result if isinstance(result, dict) else {}

def export_json_(filepath, result):
    scores = _get_scores(result)
    df = result.to_pandas()
    data = {"summary": scores, "samples": df.to_dict(orient="records")}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"结果已导出到 {filepath}")


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
    parser = argparse.ArgumentParser(description="RAGAS 生成评估 (T2Ranking queries)")
    parser.add_argument("--max-items", type=int, default=None, help="限制评估条数")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="嵌入设备 (默认 auto)")
    parser.add_argument("--k", type=int, default=4, help=f"检索 top-k (默认 4)")
    parser.add_argument("--workers", type=int, default=16, help="RAGAS 并发数 (默认 16)")
    parser.add_argument("--output", type=str, default=None, help="导出文件路径 (.csv / .json)")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "json"],
                        help="导出格式 (默认 csv)")
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
    samples = build_samples(vs, eval_queries, search_type="mmr", k=args.k)

    if not samples:
        print("没有成功生成任何样本")
        return

    # ── 评估 ──
    result_data = run_evaluation(samples, max_workers=args.workers)
    if result_data is None:
        return
    result, elapsed = result_data

    print_results(result, elapsed, len(samples))

    # ── 导出 ──
    out_path = args.output or "eval_generation_result.csv"
    if out_path.endswith(".json"):
        export_json_(out_path, result)
    else:
        export_csv_(out_path, result, samples)


if __name__ == "__main__":
    main()
