import json
import os
import sys
import tempfile
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

LANG_CODE = {"English": "en", "Russian": "ru", "Chinese": "zh", "Japanese": "ja", "German": "de", "Hebrew": "he"}
PREFERENCES = ["Overall", "Faithfulness", "Fluency", "Consistency of Style"]
SUB_PREFS = ["Faithfulness", "Fluency", "Consistency of Style"]
VALID_RESPONSES = {"A", "B", "E"}
MAX_ATTEMPTS = 3

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

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

def get_synthesized_ans(data):
    counts = {"A": 0, "B": 0, "E": 0}
    for sp in SUB_PREFS:
        v = data.get(sp)
        if v in counts:
            counts[v] += 1
    if counts["A"] > counts["B"]:
        return "A"
    if counts["B"] > counts["A"]:
        return "B"
    for sp in SUB_PREFS:
        if data.get(sp) != "E":
            return data[sp]
    return "E"

def _base_path(model_name, lc_src, lc_tgt):
    return f"./outputs/{model_name}/{lc_src}-{lc_tgt}.json"

def _split_path(model_name, lc_src, lc_tgt, split):
    return f"./outputs/{model_name}/{lc_src}-{lc_tgt}-{split}.json"

def _annotation_path(lc_src, lc_tgt, split):
    pair = f"{lc_src}-{lc_tgt}"
    return f"./annotation/{pair}/{pair}-annotation-{split}.json"

def _load_base(path):
    """Load base data as {pref: {index: entry}} where entry has src/trans1/trans2/gold/model_output."""
    if not os.path.exists(path):
        return {}
    raw = _load_json(path)
    base = {}
    for pref, records in raw.items():
        if pref == "Overall":
            continue
        base[pref] = {}
        for item in records:
            base[pref][item["index"]] = item
    return base

def _save_base(path, base_data):
    """Save base_data {pref: {index: entry}} sorted by index."""
    out = {}
    for pref in SUB_PREFS:
        if pref not in base_data:
            continue
        out[pref] = [base_data[pref][i] for i in sorted(base_data[pref])]
    _atomic_write(path, out)

def _query_api(client, model_name, prompt, temperature):
    from prompt import Get_json_result
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=8000,
    )
    content = response.choices[0].message.content
    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
    return Get_json_result(content), content, reasoning

def _query_vllm_batch(vllm_model, prompts, temperature):
    from prompt import Get_json_result
    messages = [[{"role": "user", "content": p}] for p in prompts]
    responses = vllm_model.get_response(messages, temperature)
    return [(Get_json_result(r["content"]), r["content"], r.get("reasoning_content")) for r in responses]

def _eval_pref(mode, client, mn, records, preference, base_data, base_path, src_language, tgt_language, temperature):
    from prompt import get_prompt
    if not records:
        return

    if preference not in base_data:
        base_data[preference] = {}

    if mode == "vllm":
        prompts = [get_prompt(src_language, tgt_language, preference, r["src"], r["trans1"], r["trans2"]) for r in records]
        pending = list(range(len(records)))
        preds = ["NONE"] * len(records)
        contents = [""] * len(records)
        reasonings = [None] * len(records)
        attempts = [0] * len(records)
        while pending:
            results = _query_vllm_batch(client, [prompts[i] for i in pending], temperature)
            still = []
            for i, (ans, content, reasoning) in zip(pending, results):
                contents[i], reasonings[i] = content, reasoning
                attempts[i] += 1
                if ans in VALID_RESPONSES:
                    preds[i] = ans
                elif attempts[i] < MAX_ATTEMPTS:
                    still.append(i)
                else:
                    preds[i] = "NONE"
            pending = still
        for i, r in enumerate(records):
            base_data[preference][r["index"]] = {
                "index": r["index"], "src": r["src"], "trans1": r["trans1"], "trans2": r["trans2"],
                "pred": preds[i], "gold": r.get("label"),
                "model_output": [{"content": contents[i], "reasoning": reasonings[i]}],
            }
    else:
        for r in tqdm(records, desc=preference):
            prompt = get_prompt(src_language, tgt_language, preference, r["src"], r["trans1"], r["trans2"])
            ans, content, reasoning = "NONE", "", None
            for _ in range(MAX_ATTEMPTS):
                ans, content, reasoning = _query_api(client, mn, prompt, temperature)
                if ans in VALID_RESPONSES:
                    break
            base_data[preference][r["index"]] = {
                "index": r["index"], "src": r["src"], "trans1": r["trans1"], "trans2": r["trans2"],
                "pred": ans, "gold": r.get("label"),
                "model_output": [{"content": content, "reasoning": reasoning}],
            }
            _save_base(base_path, base_data)

def run_eval(src_language, tgt_language, dataset_type, mode, model_name,
             preferences=None, temperature=0.6, api_key=None, api_url=None, model_path=None):
    if preferences is None:
        preferences = PREFERENCES

    lc_src = LANG_CODE[src_language]
    lc_tgt = LANG_CODE[tgt_language]
    base_path = _base_path(model_name, lc_src, lc_tgt)
    split_path = _split_path(model_name, lc_src, lc_tgt, dataset_type)

    # Step 1: cache hit
    if os.path.exists(split_path):
        print(f"[cache hit] {split_path}")
        return _load_json(split_path)

    # Step 2: load base data
    base_data = _load_base(base_path)

    # Step 3 & 4: load all annotation data and evaluate sub-preferences
    all_annotation_path = _annotation_path(lc_src, lc_tgt, "all")
    all_annotation = _load_json(all_annotation_path)

    if mode == "api":
        from openai import OpenAI
        client, mn = OpenAI(api_key=api_key, base_url=api_url), model_name
    else:
        from get_vllm import VllmModel
        client, mn = VllmModel(model_path=model_path), None

    sub_prefs = [p for p in preferences if p != "Overall"]
    if "Overall" in preferences:
        for sp in SUB_PREFS:
            if sp not in sub_prefs:
                sub_prefs.append(sp)

    for pref in sub_prefs:
        if pref not in all_annotation:
            print(f"[warning] '{pref}' not in annotation, skipping")
            continue
        cached = set(base_data.get(pref, {}).keys())
        records = [r for r in all_annotation[pref] if r["index"] not in cached]
        print(f"[{pref}] {len(records)} new / {len(all_annotation[pref]) - len(records)} cached")
        _eval_pref(mode, client, mn, records, pref, base_data, base_path, src_language, tgt_language, temperature)

    # Step 4: synthesize Overall
    if "Overall" in preferences:
        all_indices = {r["index"] for pref_records in all_annotation.values() for r in pref_records}
        for idx in all_indices:
            sub = {sp: base_data.get(sp, {}).get(idx, {}).get("pred", "E") for sp in SUB_PREFS}
            overall_pred = get_synthesized_ans(sub)
            # store Overall in base_data under a virtual key for scoring
            if "Overall" not in base_data:
                base_data["Overall"] = {}
            # get gold from any sub-pref entry
            gold = next((base_data[sp][idx].get("gold") for sp in SUB_PREFS if sp in base_data and idx in base_data[sp]), None)
            base_data["Overall"][idx] = {"index": idx, "pred": overall_pred, "gold": gold}

    # Step 5: normalize and save base (sub-prefs only)
    _save_base(base_path, base_data)
    print(f"[saved] {base_path}")

    # Step 6: extract split indices from annotation
    split_annotation = _load_json(_annotation_path(lc_src, lc_tgt, dataset_type))
    split_indices = {r["index"] for pref_records in split_annotation.values() for r in pref_records}

    # Build split result per-preference structure
    result = {}
    for pref in (sub_prefs + (["Overall"] if "Overall" in preferences else [])):
        if pref not in base_data:
            continue
        result[pref] = [
            base_data[pref][idx]
            for idx in sorted(base_data[pref])
            if idx in split_indices
        ]

    # Step 7: save split result
    _atomic_write(split_path, result)
    print(f"[saved] {split_path}")
    return result

def save_and_score(result, src_language, tgt_language, model_name, dataset_type="all"):
    label = f"{src_language} - {tgt_language} - {dataset_type} by {model_name}"
    print(f"\nAccuracy for {label}:")
    print("=" * 31)
    prefs_to_score = [p for p in PREFERENCES if p in result]
    for pref in prefs_to_score:
        records = result[pref]
        correct = sum(1 for r in records if r.get("pred") == r.get("gold"))
        total = len(records)
        acc = correct / total if total else 0
        print(f"{pref}: ({correct})/({total}) = {acc:.4f}")
