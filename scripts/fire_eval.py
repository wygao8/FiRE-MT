import json
import os
import sys
import tempfile
from tqdm import tqdm

LANG_CODE = {"English": "en", "Russian": "ru", "Chinese": "zh", "Japanese": "ja", "German": "de", "Hebrew": "he"}
PREFERENCES = ["Overall", "Faithfulness", "Fluency", "Consistency of Style"]
VALID_RESPONSES = {"A", "B", "E"}
MAX_ATTEMPTS = 3
SUB_PREFS = ["Faithfulness", "Fluency", "Consistency of Style"]

def _atomic_write(path, data):
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except:
        os.unlink(tmp)
        raise


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_synthesized_ans(data):
    criteria = {"A": 0, "B": 0, "E": 0}
    for sp in SUB_PREFS:
        if data.get(sp) in criteria:
            criteria[data[sp]] += 1
    if criteria["A"] > criteria["B"]:
        return "A"
    if criteria["B"] > criteria["A"]:
        return "B"
    for sp in SUB_PREFS:
        if data.get(sp) != "E":
            return data[sp]
    return "E"

def load_dataset(src_language, tgt_language, dataset_type):
    lc_src = LANG_CODE[src_language]
    lc_tgt = LANG_CODE[tgt_language]
    pair = f"{lc_src}-{lc_tgt}"
    suffix = f"-{dataset_type}" if dataset_type != "" else ""
    return _load_json(f"./annotation/{pair}/{pair}-annotation{suffix}.json")

def get_client_api(model_name, api_key, api_url):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=api_url), model_name


def get_client_vllm(model_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
    from get_vllm import VllmModel
    return VllmModel(model_path=model_path), None

def query_api(client, model_name, prompt, temperature=0.6):
    from prompt import Get_json_result
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=8000,
    )
    if not response.choices:
        return "NONE", "", None
    content = response.choices[0].message.content
    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
    return Get_json_result(content), content, reasoning

def query_vllm_batch(vllm_model, prompts, temperature=0.6):
    from prompt import Get_json_result
    messages = [[{"role": "user", "content": p}] for p in prompts]
    responses = vllm_model.get_response(messages, temperature)
    return [(Get_json_result(r["content"]), r["content"], r.get("reasoning_content")) for r in responses]


def _base_path(model_name, lc_src, lc_tgt):
    return f"./outputs/{model_name}/{lc_src}-{lc_tgt}.json"


def _split_path(model_name, lc_src, lc_tgt, split):
    return f"./outputs/{model_name}/{lc_src}-{lc_tgt}-{split}.json"


def _load_base(base_path):
    if not os.path.exists(base_path):
        return {}
    raw = _load_json(base_path)
    entries = {}
    for sp in SUB_PREFS:
        for item in raw.get(sp, []):
            idx = item["index"]
            if idx not in entries:
                entries[idx] = {"src": item["src"], "trans1": item["trans1"], "trans2": item["trans2"], "preds": {}}
            entries[idx]["preds"][sp] = {"pred": item.get("pred"), "gold": item.get("gold"), "content": item["model_output"].get("content"), "reasoning": item["model_output"].get("reasoning")}
    for item in raw.get("Overall", []):
        idx = item["index"]
        if idx not in entries:
            entries[idx] = {"src": item["src"], "trans1": item["trans1"], "trans2": item["trans2"], "preds": {}}
        entries[idx]["preds"]["Overall"] = {"pred": item.get("pred"), "gold": item.get("gold"), "model_pred": item.get("model_pred", {})}
    return entries

def _save_base(base_path, entries):
    out = {sp: [] for sp in SUB_PREFS}
    out["Overall"] = []
    for idx in sorted(entries):
        e = entries[idx]
        base = {"index": idx, "src": e["src"], "trans1": e["trans1"], "trans2": e["trans2"]}
        for sp in SUB_PREFS:
            if sp in e["preds"]:
                p = e["preds"][sp]
                out[sp].append({**base, "pred": p.get("pred"), "gold": p.get("gold"), "model_output": {"content": p.get("content"), "reasoning": p.get("reasoning")}})
        sub_preds = {sp: e["preds"].get(sp, {}).get("pred", "E") for sp in SUB_PREFS}
        sub_golds = {sp: e["preds"].get(sp, {}).get("gold") for sp in SUB_PREFS}
        overall_pred = e["preds"].get("Overall", {}).get("pred") or get_synthesized_ans(sub_preds)
        overall_gold = e["preds"].get("Overall", {}).get("gold") or get_synthesized_ans(sub_golds)
        out["Overall"].append({**base, "pred": overall_pred, "gold": overall_gold, "model_pred": {sp: sub_preds[sp] for sp in SUB_PREFS}})
    _atomic_write(base_path, out)

def _eval_records(mode, client, mn, records, preference, entries, base_path, temperature):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
    from prompt import get_prompt

    if not records:
        return

    if mode == "vllm":
        prompts = [get_prompt(r.get("src_language", ""), r.get("tgt_language", ""), preference, r["src"], r["trans1"], r["trans2"]) for r in records]
        pending = list(range(len(records)))
        preds = ["NONE"] * len(records)
        contents = [""] * len(records)
        reasonings = [None] * len(records)
        attempts = [0] * len(records)
        while pending:
            results = query_vllm_batch(client, [prompts[i] for i in pending], temperature)
            still_pending = []
            for i, (ans, content, reasoning) in zip(pending, results):
                contents[i], reasonings[i] = content, reasoning
                attempts[i] += 1
                if ans in VALID_RESPONSES:
                    preds[i] = ans
                elif attempts[i] < MAX_ATTEMPTS:
                    still_pending.append(i)
                else:
                    preds[i] = "NONE"
            pending = still_pending
        for i, r in enumerate(records):
            idx = r["index"]
            if idx not in entries:
                entries[idx] = {"src": r["src"], "trans1": r["trans1"], "trans2": r["trans2"], "preds": {}}
            entries[idx]["preds"][preference] = {"pred": preds[i], "gold": r["label"], "content": contents[i], "reasoning": reasonings[i]}
    else:
        for r in tqdm(records, desc=preference):
            prompt = get_prompt(r.get("src_language", ""), r.get("tgt_language", ""), preference, r["src"], r["trans1"], r["trans2"])
            ans, content, reasoning = "NONE", "", None
            for _ in range(MAX_ATTEMPTS):
                ans, content, reasoning = query_api(client, mn, prompt, temperature)
                if ans in VALID_RESPONSES:
                    break
            idx = r["index"]
            if idx not in entries:
                entries[idx] = {"src": r["src"], "trans1": r["trans1"], "trans2": r["trans2"], "preds": {}}
            entries[idx]["preds"][preference] = {"pred": ans, "gold": r["label"], "content": content, "reasoning": reasoning}
    
    _save_base(base_path, entries)

def run_eval(src_language, tgt_language, dataset_type, mode, model_name, preferences, temperature=0.6, api_key=None, api_url=None, model_path=None):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

    lc_src = LANG_CODE[src_language]
    lc_tgt = LANG_CODE[tgt_language]
    base_path = _base_path(model_name, lc_src, lc_tgt)
    split_path = _split_path(model_name, lc_src, lc_tgt, dataset_type)

    # Step 1: return cached split result
    if os.path.exists(split_path):
        print(f"[cache hit] {split_path}")
        return _load_json(split_path)

    # Step 2: load datasets
    raw_dataset = load_dataset(src_language, tgt_language, "")
    all_dataset = {preference:[] for preference in PREFERENCES}
    for ind, rec in enumerate(raw_dataset):
        for p in PREFERENCES:
            row = {
                "index": ind,
                "src": rec["src"],
                "trans1": rec["trans1"],
                "trans2": rec["trans2"],
                "label": rec[p]["label"],
                "level": rec[p]["level"],
                "annotation": rec[p]["annotation"],
                "model1": rec["model1"],
                "model2": rec["model2"]
            }
            all_dataset[p] += [row]
    split_dataset = load_dataset(src_language, tgt_language, dataset_type)

    # Step 3: load base cache
    entries = _load_base(base_path)

    if mode == "api":
        client, mn = get_client_api(model_name, api_key, api_url)
    else:
        client, mn = get_client_vllm(model_path)

    # Determine sub-preferences to evaluate
    sub_prefs = [p for p in preferences if p != "Overall"]
    if "Overall" in preferences:
        for sp in SUB_PREFS:
            if sp not in sub_prefs:
                sub_prefs.append(sp)

    # Step 3: process non-Overall preferences using full dataset
    for preference in sub_prefs:
        if preference not in all_dataset:
            print(f"[warning] preference '{preference}' not found in dataset, skipping")
            continue
        all_records = all_dataset[preference]
        done = {idx for idx, e in entries.items() if preference in e["preds"] and e["preds"][preference]["pred"] in ["A", "B", "E"]}
        records = [
            dict(r, src_language=src_language, tgt_language=tgt_language)
            for r in all_records if r["index"] not in done
        ]
        print(f"[{preference}] {len(records)} new / {len(all_records) - len(records)} cached")
        _eval_records(mode, client, mn, records, preference, entries, base_path, temperature)

    # Step 4: synthesize Overall pred, gold from annotation
    if "Overall" in preferences:
        overall_gold_map = {r["index"]: r["label"] for r in all_dataset.get("Overall", [])}
        for idx, e in entries.items():
            sp_flag = True
            for sp in SUB_PREFS:
                if e["preds"].get(sp, {}).get("pred", "NONE") not in ["A", "B", "E"]:
                    sp_flag = False
                    break
            if not sp_flag:
                continue
            row = {sp: e["preds"].get(sp, {}).get("pred") for sp in SUB_PREFS}
            e["preds"]["Overall"] = {
                "pred": get_synthesized_ans(row),
                "gold": overall_gold_map.get(idx),
                "model_pred": {sp: row[sp] for sp in SUB_PREFS},
            }

    # Step 5: normalize and save base
    _save_base(base_path, entries)
    print(f"[saved] {base_path}")

    # Step 6: build split result
    split_indices = {r["index"] for pref_records in split_dataset.values() for r in pref_records}
    result = _build_split_result(entries, preferences, split_indices)

    # Step 7: save split result
    _atomic_write(split_path, result)
    print(f"[saved] {split_path}")
    return result

def _build_split_result(entries, preferences, split_indices=None):
    out = {sp: [] for sp in SUB_PREFS}
    out["Overall"] = []
    for idx in sorted(entries):
        if split_indices is not None and idx not in split_indices:
            continue
        e = entries[idx]
        base = {"index": idx, "src": e["src"], "trans1": e["trans1"], "trans2": e["trans2"]}
        for sp in SUB_PREFS:
            if sp in e["preds"] and sp in preferences:
                p = e["preds"][sp]
                out[sp].append({**base, "pred": p.get("pred"), "gold": p.get("gold"), "model_output": {"content": p.get("content"), "reasoning": p.get("reasoning")}})
        if "Overall" in e["preds"]:
            op = e["preds"]["Overall"]
            out["Overall"].append({**base, "pred": op.get("pred"), "gold": op.get("gold"), "model_pred": op.get("model_pred", {})})
    return out

def save_and_score(result, src_language, tgt_language, model_name, dataset_type=""):
    label = f"{src_language} - {tgt_language} - {dataset_type} by {model_name}" if dataset_type else f"{src_language} - {tgt_language} by {model_name}"
    print(f"\nAccuracy for {label}:")
    print("=" * 31)
    overall_rows = result.get("Overall", [])
    overall_correct = sum(1 for r in overall_rows if r.get("pred") == r.get("gold"))
    overall_total = len(overall_rows)
    if overall_total:
        print(f"{'Overall':25s}: {overall_correct}/{overall_total} = {overall_correct/overall_total:.4f}")
    else:
        print("Overall: 0/0")
    for sp in SUB_PREFS:
        rows = result.get(sp, [])
        c = sum(1 for r in rows if r.get("pred") == r.get("gold"))
        t = len(rows)
        if t:
            print(f"{sp:25s}: {c}/{t} = {c/t:.4f}")
