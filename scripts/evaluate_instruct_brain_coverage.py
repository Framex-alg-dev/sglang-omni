#!/usr/bin/env python3
"""Evaluate the current Instruct model as a stateful tool-decision brain.

This deliberately does not use top-level ``tools``/``tool_choice`` fields.
Tool definitions are server-owned prompt data, model output is a strict JSON
decision plan, and a deterministic simulator returns configured tool results.
Model decisions and executor behavior are reported separately.  Cases marked
as requiring real media are never credited from their text surrogate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx


SYSTEM_PROMPT = """你是实时角色系统的任务决策大脑，不是闲聊助手。
输入会提供：当前任务状态、能力边界、可用工具定义、历史工具结果、观察事实和当前用户输入。
你必须依据这些事实规划；不得调用未声明工具，不得虚构工具已成功，不得把观察或推测冒充用户要求。
规则：
1. 用户说“别说话”时不得产生 speak/clarify；动作和表情可以调用相应工具。
2. 只采用句内纠正后的最终意图；区分修改、取消、暂停、继续、查询进度和无关闲聊。
3. 缺少执行必需参数时使用 clarify，不得猜测。
4. 取消、暂停、恢复必须绑定明确 task_id；对象不明确时澄清。
5. 工具 NOT_FOUND 不得声称已找到；FAILED 只有 retry_safe=true 才可重试；TIMEOUT_UNKNOWN 必须先 query_task_status，禁止直接重复有副作用调用。
6. 多任务必须保留已有任务，明确顺序与依赖；不要用新任务覆盖无关旧任务。
7. 视觉/观察状态中的 fact、inference、authorization 严格分开。没有策略授权和用户要求时，不得把观察自动变成操作。
8. 身份、业务和记忆只可使用输入事实或工具结果，缺参数就澄清，不得编造。

只输出一个 JSON 对象，首字符必须是 {，末字符必须是 }，禁止 Markdown 代码块和额外文字。
固定结构：
{"steps":[{"id":"s1","kind":"tool","tool":"工具名","arguments":{},"depends_on":[]}|{"id":"s1","kind":"speak","text":"实际要说的话","depends_on":[]}|{"id":"s1","kind":"clarify","question":"最小澄清问题","missing":["参数"],"depends_on":[]}|{"id":"s1","kind":"no_op","reason":"原因","depends_on":[]}],"task_updates":[{"task_id":"...","status":"..."}],"decision_summary":"一句话"}
steps 按执行顺序排列；可并行步骤使用相同 depends_on；后续步骤通过 depends_on 引用前置 id。"""


def parse_json(text: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    strict = text.lstrip().startswith("{") and text.rstrip().endswith("}")
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.S)
    try:
        value = json.loads(candidate)
    except Exception as exc:
        return None, strict, f"{type(exc).__name__}: {exc}"
    return value if isinstance(value, dict) else None, strict, None


def semantic_kind(step: dict[str, Any]) -> str | None:
    if step.get("tool"):
        return "tool"
    return step.get("kind")


def equivalent_value(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if expected == 0.3 and actual == 30:
        return True
    if expected == "21:00" and actual in {"晚上九点", "今晚九点", "9pm"}:
        return True
    if expected == "22:00" and actual in {"晚上十点", "今晚十点", "10pm"}:
        return True
    return False


def matches(step: dict[str, Any], spec: dict[str, Any]) -> bool:
    step_kind = semantic_kind(step)
    if (
        spec.get("kind") == "tool"
        and not step.get("tool")
        and step.get("kind") == spec.get("tool")
    ):
        step_kind = "tool"
    if spec.get("kind") != step_kind:
        return False
    if "tool" in spec and spec["tool"] not in {step.get("tool"), step.get("kind")}:
        return False
    actual_args = step.get("arguments") or {}
    return all(equivalent_value(actual_args.get(k), v) for k, v in spec.get("arguments", {}).items())


def judge(plan: dict[str, Any] | None, expected: dict[str, Any]) -> dict[str, bool]:
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    valid_steps = isinstance(steps, list) and all(isinstance(s, dict) for s in steps)
    checks: dict[str, bool] = {
        "parseable_json": plan is not None,
        "valid_steps": valid_steps,
    }
    if not valid_steps:
        return checks
    for index, spec in enumerate(expected.get("required", [])):
        checks[f"required_{index}"] = any(matches(step, spec) for step in steps)
    for index, spec in enumerate(expected.get("forbidden", [])):
        checks[f"forbidden_{index}"] = not any(matches(step, spec) for step in steps)
    if expected.get("exact_tool_order"):
        expected_tools = set(expected["exact_tool_order"])
        checks["tool_order"] = [
            s.get("tool") or s.get("kind")
            for s in steps
            if semantic_kind(s) == "tool" or s.get("kind") in expected_tools
        ] == expected["exact_tool_order"]
    if "max_tool_calls" in expected:
        checks["max_tool_calls"] = sum(s.get("kind") == "tool" for s in steps) <= expected["max_tool_calls"]
    if expected.get("requires_dependency"):
        checks["dependency"] = any(s.get("depends_on") for s in steps[1:])
    return checks


def contract_judge(plan: dict[str, Any] | None, strict: bool, allowed_tools: set[str]) -> dict[str, bool]:
    checks = {"strict_json": strict, "object": isinstance(plan, dict)}
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        checks["schema"] = False
        return checks
    prior_ids: set[str] = set()
    valid = True
    for step in plan["steps"]:
        if not isinstance(step, dict) or step.get("kind") not in {"tool", "speak", "clarify", "no_op"}:
            valid = False
            continue
        if step["kind"] == "tool" and (step.get("tool") not in allowed_tools or not isinstance(step.get("arguments"), dict)):
            valid = False
        if step["kind"] == "speak" and not isinstance(step.get("text"), str):
            valid = False
        if step["kind"] == "clarify" and not isinstance(step.get("question"), str):
            valid = False
        dependencies = step.get("depends_on", [])
        if not isinstance(dependencies, list) or any(dep not in prior_ids for dep in dependencies):
            valid = False
        if isinstance(step.get("id"), str):
            prior_ids.add(step["id"])
        else:
            valid = False
    checks["schema"] = valid
    return checks


async def complete(client: httpx.AsyncClient, args: argparse.Namespace, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    response = await client.post(
        args.url,
        json={
            "model": args.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"].get("content") or ""
    return text, {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "usage": payload.get("usage")}


def user_payload(case: dict[str, Any], tool_results: list[dict[str, Any]] | None = None) -> str:
    return json.dumps(
        {
            "task_state": case.get("state", {}),
            "capabilities": case.get("capabilities", {}),
            "available_tools": case.get("tools", []),
            "observation": case.get("observation"),
            "tool_results": tool_results or [],
            "user_input": case["input"],
        },
        ensure_ascii=False,
    )


async def run_case(client: httpx.AsyncClient, args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    row = {k: case.get(k) for k in ("id", "group", "variant", "input", "media_requirement")}
    if case.get("media_requirement") and not case.get("media"):
        row.update(status="pending", reason="缺少真实音频/画面资产；文本代理不计覆盖")
        return row
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload(case)},
    ]
    try:
        raw, timing = await complete(client, args, messages)
        plan, strict, parse_error = parse_json(raw)
        rounds = [{"raw": raw, "plan": plan, "strict_json": strict, "parse_error": parse_error, **timing}]
        expected = case.get("expected", {})
        checks = judge(plan, expected.get("round1", expected))
        contract_checks = contract_judge(
            plan, strict, {tool["name"] for tool in case.get("tools", [])}
        )
        simulated: list[dict[str, Any]] = []
        mock_results = case.get("mock_results", {})
        if plan and mock_results:
            for step in plan.get("steps", []):
                if step.get("kind") == "tool" and step.get("tool") in mock_results:
                    simulated.append({"step_id": step.get("id"), "tool": step.get("tool"), **mock_results[step["tool"]]})
            if simulated:
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "[模拟执行器回传]\n" + user_payload(case, simulated)},
                ])
                raw2, timing2 = await complete(client, args, messages)
                plan2, strict2, parse_error2 = parse_json(raw2)
                rounds.append({"raw": raw2, "plan": plan2, "strict_json": strict2, "parse_error": parse_error2, **timing2})
                checks.update({f"final_{k}": v for k, v in judge(plan2, expected.get("final", {})).items()})
                contract_checks.update({
                    f"final_{k}": value
                    for k, value in contract_judge(
                        plan2, strict2, {tool["name"] for tool in case.get("tools", [])}
                    ).items()
                })
        row.update(
            status="evaluated",
            rounds=rounds,
            simulated_executor={"results": simulated, "delivered": bool(simulated)},
            checks=checks,
            contract_checks=contract_checks,
            model_decision_pass=bool(checks) and all(checks.values()),
            decision_contract_pass=all(contract_checks.values()),
            system_execution_pass=all(contract_checks.values()),
        )
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}", model_decision_pass=False, system_execution_pass=False)
    return row


def coverage_status(rows: list[dict[str, Any]]) -> str:
    evaluated = [r for r in rows if r.get("status") == "evaluated"]
    if any(not r.get("model_decision_pass") and r.get("variant") == "core" for r in evaluated):
        return "未覆盖"
    if any(not r.get("model_decision_pass") for r in evaluated):
        return "部分覆盖"
    if len(evaluated) != len(rows):
        return "待验证"
    return "覆盖"


async def main(args: argparse.Namespace) -> int:
    fixture = json.loads(args.cases.read_text())
    common_tools = fixture["tools"]
    cases = [{**case, "tools": [common_tools[name] for name in case.get("tool_names", [])]} for case in fixture["cases"]]
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        async def bounded(case: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await run_case(client, args, case)
        rows = await asyncio.gather(*(bounded(case) for case in cases))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    groups = {}
    for group, items in grouped.items():
        groups[group] = {
            "model_decision_status": coverage_status(items),
            "system_execution_status": "待验证",
            "system_execution_note": "仅运行确定性模拟执行器，未证明生产编排、设备或迟到媒体行为",
            "cases": len(items),
            "evaluated": sum(i.get("status") == "evaluated" for i in items),
            "passed": sum(i.get("model_decision_pass") is True for i in items),
            "decision_contract_passed": sum(i.get("decision_contract_pass") is True for i in items),
            "simulated_execution_passed": sum(i.get("system_execution_pass") is True for i in items),
            "pending": sum(i.get("status") == "pending" for i in items),
            "failed_ids": [i["id"] for i in items if i.get("status") == "evaluated" and not i.get("model_decision_pass")],
        }
    report = {
        "scope": "current Instruct only; no Thinking comparison",
        "model": args.model,
        "endpoint": args.url,
        "method": "prompt-declared tools + strict decision JSON + deterministic simulated executor",
        "summary": {
            "cases": len(rows),
            "evaluated": sum(r.get("status") == "evaluated" for r in rows),
            "pending": sum(r.get("status") == "pending" for r in rows),
            "model_passed": sum(r.get("model_decision_pass") is True for r in rows),
            "model_failed": sum(r.get("model_decision_pass") is False for r in rows),
            "decision_contract_passed": sum(r.get("decision_contract_pass") is True for r in rows),
            "simulated_execution_passed": sum(r.get("system_execution_pass") is True for r in rows),
            "formats": dict(Counter("strict" if r.get("rounds", [{}])[-1].get("strict_json") else "non_strict" for r in rows if r.get("rounds"))),
        },
        "groups": groups,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "groups": groups}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18004/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-omni-single-gpu")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-tokens", type=int, default=700)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
