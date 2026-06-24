"""
drug_registry.py
Hebbian pharmacological association memory for NEXUS.

Models hippocampal engram formation:

  Biological process          NEXUS equivalent
  ─────────────────────────   ─────────────────────────────────────
  Perforant path input        detect_drugs() / detect_effects()
  Hebbian LTP                 co_occurrence[drug_id][effect_id] += 1
  LTP threshold               engram_threshold (default 5 exposures)
  Engram formation            build_nugget_text() → NuggetStore entry
  Synaptic pruning            weights that stay < prune_threshold removed
  Cortical consolidation      Nugget promoted to CORE after N acceptances
  Memory retrieval            catalogue_for_prompt() → text reconstruction

The drug_effect co-occurrence matrix IS the learned pharmacological
knowledge base — interpretable, auditable, growing with every training
case at zero API cost.

The 'recreate from code' step (lazy decoding):
  All internal logic operates on integer IDs.
  Text is reconstructed only when building an LLM prompt.
"""

import json
import re


# ─── Adverse effect vocabulary ────────────────────────────────────────────────

_EFFECT_PATTERNS: dict = {
    "nephrotoxicity":   r"nephrotox|renal.{0,15}(injur|fail|dysfunc|impair)|azotemia",
    "hepatotoxicity":   r"hepatotox|liver.{0,15}(injur|fail|toxicit)|hepatic.{0,15}fail|transaminase",
    "ototoxicity":      r"ototox|hearing.{0,10}(loss|impair)|tinnitus|ototoxic|cochlear",
    "cardiotoxicity":   r"cardiotox|cardiac.{0,10}(toxicit|fail|arrhythmia)|QT.{0,10}prolong|torsades",
    "neurotoxicity":    r"neurotox|encephalopathy|peripheral.{0,10}neuropathy|leukoencephalopathy",
    "hematotoxicity":   r"thrombocytopenia|leukopenia|neutropenia|pancytopenia|aplastic|agranulocytosis",
    "hypersensitivity": r"hypersensitivity|anaphyla|allergic.{0,10}reaction|angioedema",
    "dermatologic":     r"\brash\b|pruritus|urticaria|erythema|dermatitis|stevens.{0,5}johnson|toxic.{0,5}epidermal",
    "gastrointestinal": r"\bnausea\b|vomiting|diarrhea|colitis|gastrointestinal|mucositis",
    "pulmonary":        r"pneumonitis|pulmonary.{0,10}toxicit|interstitial.{0,10}lung|fibrosis",
    "endocrine":        r"hyperglycemia|hypothyroid|hyperthyroid|adrenal.{0,10}insuffic|cushingoid",
    "musculoskeletal":  r"myopathy|rhabdomyolysis|myalgia|muscle.{0,10}(weak|pain|necros)",
    "hepatic_cholestasis": r"jaundice|cholestasis|hepatitis|bilirubin",
    "renal_tubular":    r"creatinine.{0,15}(elevat|increas|rise)|proteinuria|fanconi",
    "neurological":     r"\btremor\b|seizure|ataxia|confusion|hallucination|dyskinesia",
    "thrombosis":       r"thrombosis|thromboembolism|dvt|pulmonary.{0,5}embolism|clot",
    "infusion_reaction": r"infusion.{0,10}reaction|cytokine.{0,5}release|rigor|chills",
    "electrolyte":      r"hypokalemia|hyponatremia|hypomagnesemia|hypocalcemia|hypophosphatemia",
}

# Seed known drugs — expands dynamically as new drugs are encountered
_SEED_DRUGS: set = {
    # Chemotherapy
    "cisplatin", "carboplatin", "oxaliplatin", "paclitaxel", "docetaxel",
    "doxorubicin", "epirubicin", "vincristine", "vinblastine", "bleomycin",
    "fluorouracil", "gemcitabine", "irinotecan", "etoposide", "cytarabine",
    "mercaptopurine", "methotrexate", "hydroxyurea", "mitomycin", "thioguanine",
    # Targeted / biologic
    "imatinib", "erlotinib", "gefitinib", "sorafenib", "sunitinib",
    "bortezomib", "thalidomide", "lenalidomide", "everolimus", "sirolimus",
    "rituximab", "trastuzumab", "bevacizumab", "cetuximab",
    "pembrolizumab", "nivolumab", "ipilimumab", "atezolizumab",
    "infliximab", "etanercept", "adalimumab", "tocilizumab", "abatacept",
    # Immunosuppressants
    "cyclosporine", "tacrolimus", "mycophenolate", "azathioprine",
    "hydroxychloroquine", "chloroquine",
    # Antimicrobials
    "vancomycin", "gentamicin", "tobramycin", "amikacin", "streptomycin",
    "amphotericin", "fluconazole", "voriconazole", "itraconazole",
    "clindamycin", "metronidazole", "nitrofurantoin", "dapsone",
    "isoniazid", "rifampin", "ethambutol", "pyrazinamide",
    "penicillin", "amoxicillin", "piperacillin", "ampicillin",
    "cephalosporin", "cefazolin", "ceftriaxone", "cefepime",
    "tetracycline", "doxycycline", "minocycline",
    "erythromycin", "azithromycin", "clarithromycin",
    "ciprofloxacin", "levofloxacin", "moxifloxacin",
    "trimethoprim", "sulfamethoxazole",
    # Cardiovascular
    "warfarin", "heparin", "enoxaparin", "clopidogrel", "ticagrelor",
    "rivaroxaban", "apixaban", "dabigatran",
    "digoxin", "amiodarone", "quinidine", "procainamide",
    "furosemide", "hydrochlorothiazide", "spironolactone",
    "atenolol", "metoprolol", "propranolol", "carvedilol",
    "amlodipine", "nifedipine", "diltiazem", "verapamil",
    "captopril", "enalapril", "lisinopril", "ramipril",
    "losartan", "valsartan", "irbesartan", "olmesartan",
    "simvastatin", "atorvastatin", "rosuvastatin", "pravastatin",
    # CNS
    "phenytoin", "carbamazepine", "valproate", "lamotrigine",
    "levetiracetam", "topiramate", "gabapentin", "pregabalin",
    "lithium", "clozapine", "olanzapine", "risperidone", "quetiapine",
    "aripiprazole", "haloperidol", "ziprasidone",
    "sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram",
    "venlafaxine", "duloxetine", "bupropion", "mirtazapine",
    "diazepam", "lorazepam", "alprazolam", "clonazepam", "zolpidem",
    "morphine", "oxycodone", "hydromorphone", "fentanyl", "tramadol",
    "levodopa", "carbidopa", "pramipexole", "amantadine",
    "donepezil", "memantine", "galantamine", "rivastigmine",
    # Endocrine
    "metformin", "glipizide", "insulin", "pioglitazone", "sitagliptin",
    "levothyroxine", "methimazole", "propylthiouracil",
    "prednisone", "dexamethasone", "methylprednisolone", "hydrocortisone",
    # Other
    "allopurinol", "febuxostat", "colchicine",
    "ibuprofen", "naproxen", "indomethacin", "diclofenac", "celecoxib",
    "aspirin", "acetaminophen",
    "quinine", "mefloquine",
    "interferon", "ribavirin",
}

# Drug suffix patterns — catch new drugs not in seed vocabulary
_DRUG_SUFFIX_RE = re.compile(
    r'\b([a-z]{3,}(?:mab|nib|lib|zib|cept|ximab|zumab|umab|mumab|'
    r'platin|taxel|mycin|cillin|cycline|floxacin|prazole|sartan|'
    r'pril|olol|tidine|lukast|triptan|vir|navir|ciclovir|tinib|rafenib))\b',
    re.IGNORECASE,
)
# Context: "DRUGNAME toxicity/therapy/treatment/infusion/overdose"
_DRUG_CONTEXT_RE = re.compile(
    r'\b([A-Za-z][a-z]{4,})\s+(?:toxicity|therapy|treatment|'
    r'administration|infusion|overdose|intoxication|induced|associated)\b'
)

# Blacklist: common clinical words that are NOT drugs.
# These get caught by suffix/context patterns but are anatomical, temporal,
# or generic medical terms. Run_04 showed "after", "following", "during",
# "renal" all formed false engrams. This list prevents that.
_NON_DRUG_BLACKLIST: set = {
    # Treatment strategies / generic clinical terms (not drug names)
    "combination", "therapy", "treatment", "regimen", "protocol", "monotherapy",
    "adjuvant", "neoadjuvant", "maintenance", "salvage", "induction",
    # Temporal / prepositions
    "after", "before", "during", "following", "within", "upon", "since",
    "prior", "subsequent", "concurrent", "concomitant",
    # Anatomical / organ adjectives
    "renal", "hepatic", "cardiac", "pulmonary", "cerebral", "systemic",
    "ocular", "dermal", "gastric", "enteric", "thyroid", "adrenal",
    "bilateral", "unilateral", "peripheral", "central",
    # Route of administration
    "oral", "intravenous", "topical", "inhaled", "subcutaneous",
    "intramuscular", "intraperitoneal", "intrathecal", "sublingual",
    # Severity / frequency
    "acute", "chronic", "severe", "mild", "moderate", "recurrent",
    "persistent", "transient", "progressive",
    # Disease names / symptoms / adverse effects (not drugs)
    "hepatitis", "cirrhosis", "fibrosis", "nephritis", "pneumonia",
    "colitis", "gastritis", "pancreatitis", "meningitis", "encephalitis",
    "vasculitis", "myocarditis", "pericarditis", "endocarditis",
    "thrombosis", "embolism", "infarction", "ischemia", "necrosis",
    "sepsis", "bacteremia", "viremia", "fungemia",
    "diabetes", "hypertension", "hypotension", "tachycardia",
    "bradycardia", "arrhythmia", "fibrillation",
    "seizures", "seizure", "convulsion", "tremor", "nausea", "vomiting",
    "diarrhea", "rash", "pruritus", "edema", "fatigue", "fever",
    "initial", "secondary", "primary", "standard", "typical",
    # Generic clinical nouns
    "patient", "patients", "case", "cases", "report", "treatment",
    "therapy", "dose", "doses", "level", "levels", "concentration",
    "study", "trial", "cohort", "series", "review", "analysis",
    "effect", "effects", "event", "events", "outcome", "outcomes",
    "toxicity", "reaction", "adverse", "disease", "syndrome",
    "condition", "disorder", "failure", "injury", "damage",
    "complication", "finding", "findings", "history", "status",
    # Common verbs/adjectives used as nouns
    "induced", "associated", "related", "mediated", "caused",
    "developed", "observed", "reported", "noted", "documented",
    "elevated", "increased", "decreased", "normal", "abnormal",
}

# Pre-compiled effect patterns (compiled once at import time)
_COMPILED_EFFECTS = {
    name: re.compile(pat, re.IGNORECASE)
    for name, pat in _EFFECT_PATTERNS.items()
}


class DrugRegistry:
    """
    Hebbian pharmacological association memory.

    Stores a growing vocabulary of drug names and adverse effect categories,
    and a co-occurrence weight matrix updated from every training sentence.
    No LLM calls required — all learning is statistical.

    Biological mapping
    ──────────────────
    observe()           → Hebbian LTP/LTD (neurons that fire together wire together)
    co_occurrence       → Synaptic weight matrix  [drug_id × effect_id]
    engram_threshold    → LTP induction threshold (minimum exposures for memory trace)
    engrams_ready()     → Engram consolidation check
    build_nugget_text() → Memory retrieval + cortical reconstruction (numbers → text)
    """

    def __init__(self, path: str = None,
                 engram_threshold: int = 5,
                 prune_threshold: float = 0.3):
        # ── Vocabulary (grows dynamically) ───────────────────────────────────
        self.drug_to_id:   dict = {}
        self.id_to_drug:   dict = {}
        self.effect_to_id: dict = {}
        self.id_to_effect: dict = {}

        # ── Hebbian association matrix ────────────────────────────────────────
        # co_occurrence[drug_id][effect_id] = float weight
        # Increments on ADE+effect co-occurrence, slight decay on non-ADE
        self.co_occurrence: dict = {}

        # ── Per-drug statistics ───────────────────────────────────────────────
        # drug_stats[drug_id] = {total, ade_count, not_ade_count, engram_formed}
        self.drug_stats: dict = {}

        self.engram_threshold = engram_threshold
        self.prune_threshold  = prune_threshold
        self.path = path

        # Pre-load seed vocabulary
        for drug in sorted(_SEED_DRUGS):
            self._ensure_drug(drug)
        for effect in sorted(_EFFECT_PATTERNS.keys()):
            self._ensure_effect(effect)

    # ── Vocabulary management ─────────────────────────────────────────────────

    def _ensure_drug(self, drug: str) -> int:
        drug = drug.lower().strip()
        if drug not in self.drug_to_id:
            did = len(self.drug_to_id)
            self.drug_to_id[drug]  = did
            self.id_to_drug[did]   = drug
            self.drug_stats[did]   = {
                "total": 0, "ade_count": 0,
                "not_ade_count": 0, "engram_formed": False,
            }
            self.co_occurrence[did] = {}
        return self.drug_to_id[drug]

    def _ensure_effect(self, effect: str) -> int:
        effect = effect.lower().strip()
        if effect not in self.effect_to_id:
            eid = len(self.effect_to_id)
            self.effect_to_id[effect] = eid
            self.id_to_effect[eid]    = effect
        return self.effect_to_id[effect]

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_drugs(self, text: str) -> list:
        """
        Detect drug names using three tiers:
          1. Known vocabulary (O(1) hash lookup per known drug)
          2. Pharmacological suffix patterns (catches new drug classes)
          3. Context patterns (DRUGNAME + toxicity/therapy/induced etc.)
        New drugs discovered are added to the vocabulary automatically.
        """
        t_lower = text.lower()
        found   = set()

        # Tier 1: Known vocab scan
        for drug in list(self.drug_to_id.keys()):
            if re.search(r'\b' + re.escape(drug) + r'\b', t_lower):
                found.add(drug)

        # Tier 2: Suffix-based new drug detection
        for m in _DRUG_SUFFIX_RE.finditer(text):
            candidate = m.group(1).lower()
            if len(candidate) >= 6 and candidate not in _NON_DRUG_BLACKLIST:
                self._ensure_drug(candidate)
                found.add(candidate)

        # Tier 3: Context-pattern new drug detection
        for m in _DRUG_CONTEXT_RE.finditer(text):
            candidate = m.group(1).lower()
            if len(candidate) >= 6 and candidate not in _NON_DRUG_BLACKLIST:
                self._ensure_drug(candidate)
                found.add(candidate)

        return list(found)

    def detect_effects(self, text: str) -> list:
        """Detect adverse effect categories using pre-compiled regex patterns."""
        return [eff for eff, pat in _COMPILED_EFFECTS.items() if pat.search(text)]

    # ── Hebbian learning ──────────────────────────────────────────────────────

    def observe(self, text: str, label: str):
        """
        Hebbian update from a single training case. Zero API cost.

        ADE case   → drug co-fires with effect  → LTP (+1.0 weight)
        NOT_ADE    → drug fires without effect   → mild LTD (-0.1 weight)

        Biological analog:
          Simultaneous pre- and post-synaptic activation → strengthen synapse.
          Pre-synaptic activation without post-synaptic → slight weakening.
        """
        detected_drugs   = self.detect_drugs(text)
        detected_effects = self.detect_effects(text)

        for drug in detected_drugs:
            did   = self._ensure_drug(drug)
            stats = self.drug_stats[did]
            stats["total"] += 1

            if label == "ADE":
                stats["ade_count"] += 1
                # LTP: strengthen drug → effect connections
                for effect in detected_effects:
                    eid = self._ensure_effect(effect)
                    self.co_occurrence[did][eid] = (
                        self.co_occurrence[did].get(eid, 0.0) + 1.0
                    )
            else:
                stats["not_ade_count"] += 1
                # Mild LTD: slightly weaken existing associations
                for eid in list(self.co_occurrence[did]):
                    w = self.co_occurrence[did][eid] - 0.1
                    if w <= self.prune_threshold:
                        del self.co_occurrence[did][eid]   # synaptic pruning
                    else:
                        self.co_occurrence[did][eid] = w

    def engrams_ready(self) -> list:
        """
        Return drugs that just crossed the engram formation threshold.
        Each call returns ONLY newly matured engrams (not previously reported).
        Biological analog: LTP → stable long-term memory trace (engram).
        """
        newly_formed = []
        for did, stats in self.drug_stats.items():
            if (stats["total"] >= self.engram_threshold
                    and not stats["engram_formed"]):
                stats["engram_formed"] = True
                newly_formed.append(self.id_to_drug[did])
        return newly_formed

    # ── Lazy decoding (numbers → text) ───────────────────────────────────────

    def build_nugget_text(self, drug: str) -> str | None:
        """
        Decode the numerical association profile back to human-readable text.
        This is the 'recreate from code' step — only called when building an
        LLM prompt, not during internal reasoning.

        Biological analog: Engram retrieval + hippocampal → cortical
        reconstruction. The stored trace (numbers) is decoded into language
        only when needed for communication.
        """
        drug = drug.lower()
        if drug not in self.drug_to_id:
            return None
        did   = self.drug_to_id[drug]
        stats = self.drug_stats[did]
        if stats["total"] == 0:
            return None

        ade_rate  = stats["ade_count"] / stats["total"]
        bias_word = ("HIGH" if ade_rate > 0.70
                     else "MODERATE" if ade_rate > 0.40 else "LOW")

        # Top co-occurring effects sorted by Hebbian weight
        weights    = self.co_occurrence.get(did, {})
        top_effs   = sorted(weights.items(), key=lambda x: -x[1])
        named_effs = [self.id_to_effect[eid] for eid, w in top_effs[:5] if w >= 1.0]

        parts = [
            f"{drug.title()} — ADE risk: {bias_word} "
            f"({ade_rate:.0%} ADE rate, {stats['total']} observations)."
        ]
        if named_effs:
            parts.append(f"Common adverse effects: {', '.join(named_effs)}.")
        if stats["not_ade_count"] > stats["ade_count"] * 2:
            parts.append("Frequently mentioned in non-ADE contexts — apply strict causation criteria.")

        return " ".join(parts)

    def drug_ids_for_text(self, text: str) -> list:
        """Return integer IDs of all drugs detected in text."""
        drugs = self.detect_drugs(text)
        return [self.drug_to_id[d] for d in drugs if d in self.drug_to_id]

    def effect_ids_for_text(self, text: str) -> list:
        """Return integer IDs of all adverse effects detected in text."""
        effects = self.detect_effects(text)
        return [self.effect_to_id[e] for e in effects if e in self.effect_to_id]

    def catalogue_for_prompt(self, text: str, max_drugs: int = 3) -> str:
        """
        Build a drug-profile section for injection into LLM synthesis/refine
        prompts. Only includes drugs with enough Hebbian data to be meaningful.
        Biological analog: Working memory loading from engram retrieval.
        """
        drugs = self.detect_drugs(text)
        lines = []
        shown = 0
        for drug in drugs:
            if shown >= max_drugs:
                break
            nugget = self.build_nugget_text(drug)
            if nugget:
                lines.append(f"  [{drug.upper().replace('-','_')}_PROFILE] {nugget}")
                shown += 1
        return "\n".join(lines)

    def feature_vector(self, drug_ids: list) -> dict:
        """
        Numerical ADE-risk vector for a set of drug IDs.
        Maps drug_id → ade_rate (0.0–1.0).
        Used for vector-similarity routing (future: cosine node assignment).
        """
        vec = {}
        for did in drug_ids:
            stats = self.drug_stats.get(did)
            if stats and stats["total"] > 0:
                vec[did] = stats["ade_count"] / stats["total"]
        return vec

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary_stats(self) -> dict:
        engrams = [
            self.id_to_drug[did]
            for did, s in self.drug_stats.items()
            if s["engram_formed"]
        ]
        top_ade = sorted(
            [
                (self.id_to_drug[did],
                 round(s["ade_count"] / max(1, s["total"]), 3),
                 s["total"])
                for did, s in self.drug_stats.items()
                if s["total"] >= 3
            ],
            key=lambda x: -x[1],
        )[:10]
        return {
            "vocab_size":          len(self.drug_to_id),
            "effects_tracked":     len(self.effect_to_id),
            "engrams_formed":      len(engrams),
            "engram_drugs":        engrams[:20],
            "top_ade_rate_drugs":  top_ade,
        }

    def print_report(self):
        s = self.summary_stats()
        print("\n=== Drug Registry (Hebbian Memory) ===")
        print(f"  Vocabulary : {s['vocab_size']} drugs  |  "
              f"{s['effects_tracked']} effect categories")
        print(f"  Engrams    : {s['engrams_formed']} formed "
              f"(threshold={self.engram_threshold})")
        if s["engram_drugs"]:
            print(f"  Drugs with engrams: {', '.join(s['engram_drugs'][:10])}")
        if s["top_ade_rate_drugs"]:
            print("  Highest ADE-rate drugs (min 3 obs):")
            for drug, rate, n in s["top_ade_rate_drugs"][:5]:
                print(f"    {drug}: {rate:.0%} ADE ({n} obs)")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "version":          2,
                "engram_threshold": self.engram_threshold,
                "prune_threshold":  self.prune_threshold,
                "drug_to_id":       self.drug_to_id,
                "id_to_drug":       {str(k): v for k, v in self.id_to_drug.items()},
                "effect_to_id":     self.effect_to_id,
                "id_to_effect":     {str(k): v for k, v in self.id_to_effect.items()},
                "co_occurrence":    {
                    str(did): {str(eid): w for eid, w in effs.items()}
                    for did, effs in self.co_occurrence.items()
                },
                "drug_stats":       {str(k): v for k, v in self.drug_stats.items()},
            }, f, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "DrugRegistry":
        try:
            with open(path) as f:
                data = json.load(f)
            reg = cls(
                path=path,
                engram_threshold=data.get("engram_threshold", 5),
                prune_threshold=data.get("prune_threshold", 0.3),
            )
            reg.drug_to_id   = data["drug_to_id"]
            reg.id_to_drug   = {int(k): v for k, v in data["id_to_drug"].items()}
            reg.effect_to_id = data["effect_to_id"]
            reg.id_to_effect = {int(k): v for k, v in data["id_to_effect"].items()}
            reg.co_occurrence = {
                int(dk): {int(ek): w for ek, w in effs.items()}
                for dk, effs in data["co_occurrence"].items()
            }
            reg.drug_stats = {int(k): v for k, v in data["drug_stats"].items()}
            return reg
        except FileNotFoundError:
            inst = cls(path=path, **kwargs)
            inst.save(path)
            return inst
