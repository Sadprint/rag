"""
RAGAS 指标评估 — 从样本 JSON 加载，可反复运行调整指标组合

用法:
  python evaluate_metrics.py                                   默认加载 eval_samples.json
  python evaluate_metrics.py --input samples.json               指定样本文件
  python evaluate_metrics.py --metrics faith,ctx_precision,ctx_util
  python evaluate_metrics.py --output result.csv --workers 16
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import types

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["TQDM_DISABLE"] = "1"

# ── onnxruntime mock ──
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

_register("onnxruntime")
_register("onnxruntime.capi")
_register("onnxruntime.capi._pybind_state")

# ── vertexai mock ──
from langchain_core.language_models.chat_models import BaseChatModel as _BaseChatModel
_mock_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
_mock_vertexai.__package__ = "langchain_community.chat_models"

class _FakeChatVertexAI(_BaseChatModel):
    def _generate(self, *a, **kw): pass
    def _llm_type(self): return "fake"

_mock_vertexai.ChatVertexAI = _FakeChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _mock_vertexai

# Windows GBK → UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ContextPrecision, ContextUtilization

import config

# ── 所有可用指标 ──
METRIC_REGISTRY = {
    "faith":          ("faithfulness",        lambda j: Faithfulness(llm=j)),
    "ctx_precision":  ("context_precision",   lambda j: ContextPrecision(llm=j)),
    "ctx_util":       ("context_utilization", lambda j: ContextUtilization(llm=j)),
}


def load_samples(filepath):
    """从 JSON 加载样本，返回 (List[SingleTurnSample], meta)"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data["samples"]
    meta = data.get("meta", {})
    samples = []
    for item in raw:
        samples.append(SingleTurnSample(
            user_input=item["user_input"],
            response=item["response"],
            retrieved_contexts=item.get("retrieved_contexts", []),
            reference=item.get("reference"),
        ))
    return samples, meta


def run_evaluation(samples, metric_names, max_workers=16, verbose=True):
    """用 RAGAS 并发评估样本"""
    if not samples:
        return None

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.ZHIPUAI_API_KEY,
        base_url=config.ZHIPUAI_BASE_URL,
        temperature=0.01,
    )
    judge = LangchainLLMWrapper(llm)

    metrics = [METRIC_REGISTRY[name][1](judge) for name in metric_names]
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
        print(f"  评估完成，耗时 {elapsed:.1f}s ({len(samples) / elapsed:.1f} 样本/s, {max_workers} 并发)")

    return result, elapsed


def _compute_means(scores_list):
    """聚合逐条分数为均值"""
    if not scores_list:
        return {}
    keys = scores_list[0].keys()
    means = {}
    for k in keys:
        vals = []
        errors = 0
        for s in scores_list:
            v = s.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                errors += 1
                continue
            vals.append(v)
        means[k] = sum(vals) / len(vals) if vals else float("nan")
        means[f"{k}_errors"] = errors
    return means


def print_results(result, elapsed, num_samples, metric_names):
    """格式化输出"""
    metric_labels = [METRIC_REGISTRY[n][0] for n in metric_names]

    print()
    print("=" * 58)
    print("  RAGAS 生成评估结果")
    print("=" * 58)
    print(f"  样本数:        {num_samples}")
    print(f"  评估耗时:      {elapsed:.1f}s")
    print(f"  judge:         {config.LLM_MODEL}")
    print()

    means = _compute_means(result.scores)
    print(f"  {'指标':25s} {'均值':>8s}  {'错误':>5s}")
    print(f"  {'─' * 33}  {'─' * 5}")
    for name in metric_labels:
        score = means.get(name, float("nan"))
        errors = means.get(f"{name}_errors", 0)
        print(f"  {name:25s} {score:8.4f}  {errors:5d}")
    print("=" * 58)

    # 最低 faithfulness 的 5 条
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

    desc = {
        "faithfulness":        "答案中有多少事实可在检索文档中找到依据（防幻觉）",

        "context_precision":   "标注的相关文档是否排在检索结果前列",
        "context_utilization": "检索文档中有多少信息被答案所使用",
    }
    print("  指标说明:")
    for name in metric_labels:
        print(f"    {name:25s} — {desc.get(name, '')}")


def export_csv_(filepath, result):
    """导出含逐条分数的 CSV"""
    df = result.to_pandas()
    if "retrieved_contexts" in df.columns:
        df["retrieved_contexts"] = df["retrieved_contexts"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else str(x)
        )
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"结果已导出到 {filepath} ({len(df)} 行 × {len(df.columns)} 列)")


def export_json_(filepath, result):
    means = _compute_means(result.scores)
    df = result.to_pandas()
    data = {"summary": means, "samples": df.to_dict(orient="records")}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"结果已导出到 {filepath}")


def main():
    parser = argparse.ArgumentParser(description="RAGAS 指标评估 (从样本 JSON 加载)")
    parser.add_argument("--input", type=str, default="results/eval_samples.json",
                        help="样本 JSON 文件 (默认 results/eval_samples.json)")
    parser.add_argument("--output", type=str, default="results/eval_metrics_result.csv",
                        help="导出文件路径 (.csv / .json)")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "json"],
                        help="导出格式 (默认 csv)")
    parser.add_argument("--workers", type=int, default=16, help="RAGAS 并发数 (默认 16)")
    parser.add_argument("--metrics", type=str,
                        default="faith,ctx_precision,ctx_util",
                        help="指标列表，逗号分隔。可选: faith, ctx_precision, ctx_util")
    args = parser.parse_args()

    metric_names = [m.strip() for m in args.metrics.split(",")]
    for m in metric_names:
        if m not in METRIC_REGISTRY:
            print(f"未知指标: {m}")
            print(f"可选: {', '.join(METRIC_REGISTRY.keys())}")
            return

    # ── 加载样本 ──
    print(f"加载样本: {args.input}")
    samples, meta = load_samples(args.input)
    print(f"  样本数: {len(samples)}")
    print(f"  参数: k={meta.get('k', '?')}, search_type={meta.get('search_type', '?')}")

    # ── 评估 ──
    result_data = run_evaluation(samples, metric_names, max_workers=args.workers)
    if result_data is None:
        return
    result, elapsed = result_data

    print_results(result, elapsed, len(samples), metric_names)

    # ── 导出 ──
    out_path = args.output
    if out_path.endswith(".json"):
        export_json_(out_path, result)
    else:
        export_csv_(out_path, result)


if __name__ == "__main__":
    main()
