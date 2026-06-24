"""
features.py
Pure-regex feature extraction for NEXUS. No ML, no API calls.

These 11 boolean features are the ONLY inputs the tree router and the
synthesis LLM are allowed to use in trigger_condition expressions.
"""

import re

FEATURE_NAMES = [
    'has_induced', 'has_associated', 'has_toxicity', 'has_adverse',
    'has_developed', 'has_following', 'has_reaction', 'has_report',
    'has_negation', 'has_short', 'has_drug_name',
]

_DRUG_RE = re.compile(
    # Broad pharmacological suffix pattern catches drug classes not explicitly listed
    r'\b(?:'
    # Chemotherapy
    r'cisplatin|carboplatin|oxaliplatin|paclitaxel|docetaxel|'
    r'doxorubicin|epirubicin|vincristine|vinblastine|bleomycin|'
    r'fluorouracil|gemcitabine|irinotecan|etoposide|cytarabine|'
    r'mercaptopurine|methotrexate|hydroxyurea|mitomycin|thioguanine|'
    # Targeted / biologic
    r'imatinib|erlotinib|gefitinib|sorafenib|sunitinib|bortezomib|'
    r'thalidomide|lenalidomide|everolimus|sirolimus|'
    r'rituximab|trastuzumab|bevacizumab|cetuximab|'
    r'pembrolizumab|nivolumab|ipilimumab|atezolizumab|'
    r'infliximab|etanercept|adalimumab|tocilizumab|'
    # Immunosuppressants
    r'cyclosporine|tacrolimus|mycophenolate|azathioprine|'
    r'hydroxychloroquine|chloroquine|'
    # Antibiotics / antivirals
    r'vancomycin|gentamicin|tobramycin|amikacin|streptomycin|'
    r'amphotericin|fluconazole|voriconazole|itraconazole|'
    r'clindamycin|metronidazole|nitrofurantoin|dapsone|'
    r'isoniazid|rifampin|ethambutol|pyrazinamide|'
    r'penicillin|amoxicillin|piperacillin|ampicillin|'
    r'ceftriaxone|cefepime|cefazolin|'
    r'tetracycline|doxycycline|minocycline|'
    r'erythromycin|azithromycin|clarithromycin|'
    r'ciprofloxacin|levofloxacin|moxifloxacin|'
    r'trimethoprim|sulfamethoxazole|'
    # Cardiovascular
    r'warfarin|heparin|enoxaparin|clopidogrel|ticagrelor|'
    r'rivaroxaban|apixaban|dabigatran|'
    r'digoxin|amiodarone|quinidine|'
    r'furosemide|hydrochlorothiazide|spironolactone|'
    r'atenolol|metoprolol|propranolol|carvedilol|'
    r'amlodipine|nifedipine|diltiazem|verapamil|'
    r'captopril|enalapril|lisinopril|ramipril|'
    r'losartan|valsartan|irbesartan|'
    r'simvastatin|atorvastatin|rosuvastatin|pravastatin|'
    # CNS
    r'phenytoin|carbamazepine|valproate|lamotrigine|'
    r'levetiracetam|topiramate|gabapentin|pregabalin|'
    r'lithium|clozapine|olanzapine|risperidone|quetiapine|'
    r'aripiprazole|haloperidol|'
    r'sertraline|fluoxetine|paroxetine|citalopram|escitalopram|'
    r'venlafaxine|duloxetine|bupropion|mirtazapine|'
    r'diazepam|lorazepam|alprazolam|clonazepam|zolpidem|'
    r'morphine|oxycodone|hydromorphone|fentanyl|tramadol|'
    r'levodopa|carbidopa|pramipexole|amantadine|'
    r'donepezil|memantine|galantamine|rivastigmine|'
    # Endocrine / other
    r'metformin|glipizide|insulin|pioglitazone|'
    r'levothyroxine|methimazole|propylthiouracil|'
    r'prednisone|dexamethasone|methylprednisolone|hydrocortisone|'
    r'allopurinol|febuxostat|colchicine|'
    r'ibuprofen|naproxen|indomethacin|diclofenac|celecoxib|'
    r'aspirin|acetaminophen|interferon|ribavirin|quinine|mefloquine'
    r')\b'
    r'|'
    # Generic pharmacological suffixes catch unlisted drugs (e.g. pembrolizumab, ruxolitinib)
    r'\b[a-z]{3,}(?:mab|nib|tinib|rafenib|cept|zumab|ximab|platin|taxel|'
    r'mycin|cillin|cycline|floxacin|prazole|sartan|pril|olol|vir|navir)\b',
    re.IGNORECASE,
)
_INDUCED_RE = re.compile(r'\b\w+-induced\b')
_ASSOCIATED_RE = re.compile(r'\b\w+-associated\b')
_NEGATION_RE = re.compile(r'\bno\b|\bnot\b|\bwithout\b|\bdenied\b')


def features(text: str) -> dict:
    """Return the 11 boolean features for a given input sentence."""
    t = text.lower()
    return {
        'has_induced':    bool(_INDUCED_RE.search(t)),
        'has_associated': bool(_ASSOCIATED_RE.search(t)),
        'has_toxicity':   'toxicity' in t or 'toxic' in t,
        'has_adverse':    'adverse' in t or 'side effect' in t,
        'has_developed':  'developed' in t or 'developed after' in t,
        'has_following':  'following' in t or 'after' in t,
        'has_reaction':   'reaction' in t or 'hypersensitivity' in t,
        'has_report':     'report' in t or 'case' in t,
        'has_negation':   bool(_NEGATION_RE.search(t)),
        'has_short':      len(text) < 80,
        'has_drug_name':  bool(_DRUG_RE.search(t)),
    }


def safe_eval_condition(trigger_condition: str, feats: dict) -> bool:
    """
    Evaluate a trigger_condition string against a features dict, but ONLY
    if it uses the allowed feature names and safe boolean operators.

    This is the gate that stops the synthesis LLM from injecting arbitrary
    code via trigger_condition. Returns False (never matches) if validation
    fails, rather than raising, so a bad graft is silently unroutable
    rather than crashing the pipeline.
    """
    if not trigger_condition or not trigger_condition.strip():
        return False

    # Tokenize and check every identifier-like token is either an allowed
    # feature name or a safe boolean keyword.
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', trigger_condition)
    allowed = set(FEATURE_NAMES) | {'and', 'or', 'not', 'True', 'False'}
    for tok in tokens:
        if tok not in allowed:
            return False

    # Disallow anything that isn't a name, boolean operator, parens, or
    # whitespace -- e.g. blocks "()", "__import__", attribute access, etc.
    if not re.fullmatch(r'[A-Za-z0-9_\s()]*', trigger_condition):
        return False

    try:
        # eval() here is safe ONLY because we've already restricted the
        # expression to whitelisted identifiers + and/or/not/parens above.
        # feats dict supplies the only names in scope; builtins are blocked.
        return bool(eval(trigger_condition, {"__builtins__": {}}, dict(feats)))
    except Exception:
        return False


def is_valid_condition(trigger_condition: str) -> bool:
    """
    Validate a candidate trigger_condition string without evaluating it
    against real data. Used at graft-proposal time (spec step 5: VALIDATE).
    """
    dummy_feats = {name: False for name in FEATURE_NAMES}
    if not trigger_condition or not trigger_condition.strip():
        return False
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', trigger_condition)
    allowed = set(FEATURE_NAMES) | {'and', 'or', 'not', 'True', 'False'}
    if any(tok not in allowed for tok in tokens):
        return False
    if not re.fullmatch(r'[A-Za-z0-9_\s()]*', trigger_condition):
        return False
    try:
        eval(trigger_condition, {"__builtins__": {}}, dummy_feats)
        return True
    except Exception:
        return False
