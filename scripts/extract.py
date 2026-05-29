import json

src_lang, tgt_lang = "ru", "zh"
path = f"../annotation/{src_lang}-{tgt_lang}/{src_lang}-{tgt_lang}-annotation.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

def extract_from_data(data, mode = 'all'):

    labels = ["A", "B", "E"]
    if mode == "tied":
        labels = ["E"]
    elif mode == "ranked":
        labels = ["A", "B"]

    preferences = ["Faithfulness", "Fluency", "Consistency of Style", "Overall"]
    result = {p: [] for p in preferences}

    for index, item in enumerate(data):
        for pref in preferences:
            p = item.get(pref, {})
            if p.get("label") in labels:
                result[pref].append({
                    "index": index,
                    "src": item["src"],
                    "trans1": item["trans1"],
                    "trans2": item["trans2"],
                    "label": p["label"],
                    "level": p["level"],
                    "annotation": p["annotation"],
                    "model1": item["model1"],
                    "model2": item["model2"],
                })

    out_path = f"../annotation/{src_lang}-{tgt_lang}/{src_lang}-{tgt_lang}-annotation-{mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    for p in preferences:
        print(f"{p}: {len(result[p])} entries")

extract_from_data(data, "all")
extract_from_data(data, "tied")
extract_from_data(data, "ranked")
