import re
PROMPT_HEADER='''You are a translation evaluator. Given a triple ([source], [translation A], [translation B]), where [translation A] and [translation B] are two translation candidates. Please compare two translation candidates based on the source language text under the Given Preference with Note and making a relative evaluation of their quality. Please answer based on analysis and write the analysis and result in the format {"analysis": ###TEMPLATE###, "result":...}.\n\nThe marked options are divided into three categories, with the following specific meanings:\n- A: The quality of [A] is higher than the quality of [B]\n- B: The quality of [B] is higher than the quality of [A]\n- E: The quality of [A] is equivalent to the quality of [B], and it is impossible to distinguish the superiority or inferiority\nIf both translations contain errors, please determine which translation has more significant errors (choose A or B), or if both have errors of similar severity (choose E).\n\n'''

PROMPT_DESCRIPTION = '''### Preference ###\n###DESCRIPTION###\n'''
PROMPT_NOTE = '''### Note ###\n###NOTE###\n'''
PROMPT_END = '''### Translation Evaluation ####\n#SRC_LANGUAGE# Source: ###SRC_TEXT###\n#TGT_LANGUAGE# Translation A: ###TRANSLATION_A###\n#TGT_LANGUAGE# Translation B: ###TRANSLATION_B###\nPreference: ###PREFERENCE_END###\n'''

PREFERENCE_DICT={
    "Overall":{
        "template":"...",
        "end":"Overall - Combine [source] to comprehensively compare [A] and [B].",
        "note":{"all": 
'''Regarding Overall, Please consider the following points:
- If the text content is exactly the same, prefer using punctuation of the target language.
- If the original text is untranslatable (e.g. html, url, emoji ...), the translation should be copied directly.
- Keep emojis
- If a translation is empty, the quality is the lowest. If two translation are both empty, the quality is the same.
- If a translation is not in #TGT_LANGUAGE#, the quality is the lowest.
'''}
},

    "Faithfulness":{
        "template":f'{{"Accuracy of Information":..., "Accuracy of Named Entities":...}}',
        "description":
        {
            "all": 
'''Faithfulness (reflects the severity of hallucinations in the translation model)
    A. Accuracy of Information: Faithful to the original text, with no missing, incorrect, or added information.
    B. Accuracy of Named Entities: Names of people, places, organizations, and specialized terms, as well as times, quantities, currency, ratios, and other specifics are accurately translated.
'''
        },
        "note":{"all": 
'''Regarding Faithfulness, Please consider the following points:
- Unknown or unverifiable people and place names do not affect the translation; well-known proper nouns should refer to standard translations.
- Core words, Important sentences and grammatical structures must be translated.
- Adding, deleting, or modifying information must not alter the original meaning.
- Sentence tense should be consistent.
- Do **not** consider Fluency or Consistency of Style or any other preferences that are unrelated to Faithfulness.
- If the original text is untranslatable (e.g. html, url, emoji ...), the translation should be copied directly.
- Keeps emoji
- If a translation is empty, the quality is the lowest. If two translation are both empty, the quality is the same.
- If a translation is not in #TGT_LANGUAGE#, the quality is the lowest.
'''
        },
        "end":"Faithfulness - Combined with [source], description of Faithfulness and Note to compare [A] and [B] under Faithfulness only."
    },
    "Fluency":{
        "template":f'{{"Lexical Quality":..., "Syntactic Quality":..., "Punctuation":..., "Untranslated":...}}',
        "description":
        {
            "all":
'''Fluency (reflects the language capabilities of the translation model)
    A. Lexical Quality: Proper word choice, parts of speech, spelling, and capitalization.
    B. Syntactic Quality: Correct sentence structure, word order.
    C. Punctuation: Punctuation incorrect according to target language conventions. Missing mark from a set of paired punctuation marks, such as a missing parenthesis or quote mark.
    D. Untranslated: untranslated names of people or places
'''
        },
        "note":{
            "all": 
'''Regarding Fluency, Please consider the following points:
- Smooth sentences are prioritized. Pay attention to Inappropriate collocation, which making the sentence awkward (but not affecting understanding).
- Adding, deleting, or modifying information should aid in understanding the sentence.
- Use the punctuation marks of the target language
- Do **not** consider Faithfulness or Consistency of Style or any other preferences that are unrelated to Fluency.
- If the original text is untranslatable (e.g. html, url, emoji ...), the translation should be copied directly.
- Keeps emoji
- If a translation is empty, the quality is the lowest. If two translation are both empty, the quality is the same.
- If a translation is not in #TGT_LANGUAGE#, the quality is the lowest.
'''
        },

        "end":"Fluency - Combined with [source], description of Fluency and Note to compare [A] and [B] under Fluency only."
    },

    "Consistency of Style":{
        "template":f'{{"Tone Matching":..., "Emotional Preservation":..., "Writing Style":}}',
        "description":
        {
        "all": 
'''Consistency of Style (reflects the model's ability to transfer style across languages)
    A. Tone Matching: The translated text’s tone should match the source, whether academic, technical, or conversational.
    B. Emotional Preservation: The translation should convey the original text's emotional tone or mood, whether positive, negative or neutral. The translation should maintain the original.
    C. Writing Style: The translation should reflect the original style, whether concise and direct or detailed and thorough.
'''
        },
        "note":{
        "all": 
'''Regarding Consistency of Style, Please consider the following points:
- Do **not** consider Faithfulness or Fluency or any other preferences that are unrelated to Consistency of Style.
- If the original text is untranslatable (e.g. html, url, emoji ...), the translation should be copied directly.
- Keeps emojis
- If a translation is empty, the quality is the lowest. If two translation are both empty, the quality is the same.
- If a translation is not in #TGT_LANGUAGE#, the quality is the lowest.
'''
        },
        "end":"Consistency of Style - Combined with [source], description of Consistency of Style and Note to compare [A] and [B] under Consistency of Style only."
    },
    "Locale Convention":{
        "template": f'{{"Formatting Conventions":..., "Naming and Title Conventions":..., "Orthographic and Typographic Conventions":...}}',
        "description":{
            "all": 
'''Locale Convention in terms of the following aspects:
    A. Formatting Conventions: Dates, times, numbers, currency, percentages, and measurement expressions (e.g., distance, temperature, and weight) should follow target-language conventions and common usage.
    B. Naming and Title Conventions: Names of organizations, institutions, job titles, honorifics, and other conventional designations should be rendered in a way that matches common target-language conventions.
    C. Orthographic and Typographic Conventions: Punctuation, quotation marks, book/article title marks, spacing, and other writing conventions should conform to target-language norms.
'''
        },
        "end":"Locale Convention - Combined with [source], description of Locale Convention to compare [A] and [B] under Locale Convention only."
    }
}

def get_prompt(src_language, tgt_language, preference, src_text, translation1, translation2):
    prompt = ""
    preference = preference
    prompt = PROMPT_HEADER
    if "description" in PREFERENCE_DICT[preference] and "all" in PREFERENCE_DICT[preference]["description"]:
        prompt += PROMPT_DESCRIPTION.replace("###DESCRIPTION###",PREFERENCE_DICT[preference]["description"]["all"])
    if "note" in PREFERENCE_DICT[preference] and "all" in PREFERENCE_DICT[preference]["note"]:
        prompt += PROMPT_NOTE.replace("###NOTE###", PREFERENCE_DICT[preference]["note"]["all"])
    
    prompt += PROMPT_END
    prompt = prompt.replace("#SRC_LANGUAGE#", src_language).replace("#TGT_LANGUAGE#", tgt_language).replace("###PREFERENCE_END###", PREFERENCE_DICT[preference]["end"]).replace("###TEMPLATE###",PREFERENCE_DICT[preference]["template"])
    return prompt.replace("###SRC_TEXT###", src_text).replace('###TRANSLATION_A###', translation1).replace('###TRANSLATION_B###', translation2)

def Get_json_result(js_str:str):
    js_str = js_str.lower()
    rA = re.findall(r'["\\]*result["\\]*\s*:[\\\s"{]*a', js_str)
    if len(rA) > 0:
        eA = list(re.finditer(r'["\\]*result["\\]*\s*:[\\\s"{]*a', js_str))[-1].end()
    else:
        eA = -1

    rB = re.findall(r'["\\]*result["\\]*\s*:[\\\s"{]*b', js_str)
    if len(rB) > 0:
        eB = list(re.finditer(r'["\\]*result["\\]*\s*:[\\\s"{]*b', js_str))[-1].end()
    else:
        eB = -1
    
    rE = re.findall(r'["\\]*result["\\]*\s*:[\\\s"{]*e', js_str)
    if len(rE) > 0:
        eE = list(re.finditer(r'["\\]*result["\\]*\s*:[\\\s"{]*e', js_str))[-1].end()
    else:
        eE = -1

    i = -1
    ans = "###NONE###"+js_str

    if eA > i:
        i = eA
        ans = "A"
    if eB > i:
        i = eB
        ans = "B"
    if eE > i:
        i = eE
        ans = "E"
    return ans