# NEXUS Research Diary
**Principal Investigator:** Yasir El-Sherif, MD — Northwell Health
**Collaborator:** Claude (Anthropic) — computational architecture and implementation
**Started:** June 23, 2026
**Document type:** Living scientific diary — narrative of discovery

---

> *This is not a technical manual. It is the record of a scientific mind learning how minds learn.*
>
> *The question we are trying to answer: can a machine learning system develop — not just train — in a way that mirrors the developmental biology of the brain? And if so, what does that teach us about both machines and minds?*

---

## Entry 1 — June 23, 2026: The First Question

**What we started with:**
A single observation: large language models (LLMs) are powerful but static. They are trained once, deployed, and never truly learn from new experience in a task-specific way. They are experts who forgot how to grow.

**The hypothesis:**
If we build a system around an LLM that can route tasks to specialist agents, and if those specialists can accumulate structured memory, then the system might learn to classify clinical text better over time — not by retraining weights, but by building better cognitive scaffolding around a frozen LLM core.

**The biological frame:**
A cortical column in the human brain does not rewrite its DNA when it learns something new. Instead, it strengthens certain synaptic connections, consolidates patterns into long-term memory, and passes error signals forward to neighboring columns. Can we build something that works the same way?

**The first experiment:**
A single LLM call per ADE classification sentence. No memory. No structure. No feedback.

**Result:**
The system classified correctly on approximately the first try — but it never improved. Each sentence was processed from scratch as if it had never seen the domain before. Performance was decent (baseline ~0.70 F1) but flat.

**Implication:**
A single LLM call is a single neuron. Useful. Not a brain.

---

## Entry 2 — June 23–24, 2026: Building Structure

**The progression:**
Over the next 24 hours we built, iteratively, the layered components that would eventually become the NEXUS cortex:

- **RAG index** (hippocampal retrieval): similar examples surface during classification, giving the LLM context it cannot generate from parameters alone. Immediate F1 improvement.

- **Parallel expert routes** (cortical streams): four simultaneous LLM calls — causation, negation, drug effect, context — vote on the final label. Mimics the dorsal/ventral processing streams of visual cortex (Ungerleider & Mishkin 1982). Recall improved sharply because diverse failure modes were no longer coupled.

- **Self-growing decision tree** (v3.04): a tree of specialist NexusNodes that grow based on error cluster analysis. Each node has its own prompt, memory, and specialization trigger. The system could propose, evaluate, and add new specialist nodes automatically.

**Peak performance:**
v3.04 at Round 1: **F1 = 0.9412.** The highest score in the project.

**What happened next:**
The system grew. New nodes were grafted. By Round 20, there were 13 nodes and F1 had collapsed to 0.7308.

**First major finding ★:**
*The system peaked at its seed configuration and degraded with every addition.*

This is a counter-intuitive and publishable result. Most machine learning systems improve with more parameters, more layers, more data. NEXUS v3.04 degraded with more structure. The tree grew itself into incoherence.

**Root cause (FM-1):**
FIFO routing — first matching node wins. New nodes with overlapping triggers silently stole cases from mature, calibrated nodes. The expert who had learned 100 examples got replaced by a naive novice who happened to fire first.

**Biological analogy:**
A newborn neuron in adult neurogenesis (Eriksson 1998) that integrates too early — before it has formed proper synaptic connections — can disrupt the circuit it joins. Not all neurogenesis is beneficial. The question is not "can you add a neuron?" but "should you add this neuron now?"

---

## Entry 3 — June 25–26, 2026: The Cortical Architecture

**The redesign hypothesis:**
If FIFO routing causes fragmentation, replace it with biology's own solution: **competitive routing by specificity.** The most specialized matching column wins. Generalist columns fire only when no specialist applies. This is how the neocortex works — hierarchical selectivity (Mountcastle 1997).

**nexus_cortex_v1.py — the full biological architecture:**

We implemented, in working code, six mechanisms from the computational neuroscience literature:

1. **BCM Theory** (Bienenstock, Cooper, Munro 1982): The sliding modification threshold θ_M rises when a neuron fires often and falls when it fires rarely. Frequent activation → LTP. Rare activation → LTD. The modification threshold itself is the memory of past activity.

2. **Critical Period Plasticity** (Wiesel & Hubel 1963; Hensch 2004): A developmental window during which the cortex is maximally plastic, after which structure consolidates. We implemented three phases: EMBRYONIC (permissive growth), DEVELOPMENTAL (selective growth), CONSOLIDATION (pruning dominant).

3. **Homeostatic Plasticity** (Turrigiano et al. 1998): The system continuously calibrates a FiringThreshold to maintain a target recall setpoint. Zero LLM cost — uses cached score tuples from evaluation. Biological analogy: synaptic scaling to prevent runaway excitation.

4. **Competitive Routing** (Rumelhart & Zipser 1985): Winner-take-all selection by specificity. Solved FM-1 completely.

5. **Neurogenesis + Apoptosis** (Eriksson et al. 1998): New columns are proposed, probed, and either survive or are rolled back immediately if they hurt global F1. Columns that never activate are pruned with their memory bequeathed to the parent.

6. **Engram Consolidation** (Semon 1904; Josselyn 2015): Error clusters (EnggramClusters) detected by cosine similarity → consolidated into MemoryTraces → inherited by genesis columns and parent on apoptosis.

**What worked:**
- BCM dynamics fired correctly throughout: ROOT earned 20 consecutive LTP events
- Rollback mechanism (FM-7) correctly rejected 8/10 genesis proposals
- First apoptosis at R15: COL_TEMPORAL_OBSERVATION (0 activations, 1 trace bequeathed to ROOT)
- Homeostatic calibration maintained stable F1 band without cliff collapse

**What failed:**
- Peak F1 at R3 (0.8182), then stable oscillation around 0.75–0.81 through R20
- Net learning over 20 rounds: essentially zero (R1=0.778, R20=0.759)
- The embryo stabilized without developing

---

## Entry 4 — June 26, 2026: The Three Missing Teachers

**The core finding from v1.0-cortex Run 2:**

*The system had memory but no lessons. Structure but no pedagogy. Plasticity but no guidance.*

After R3, every specialist column stayed in deep LTD (rehearsal_weight=0.30). They received error cases but couldn't learn from them because raw error cases contain only "you were wrong" — not "here is why, and here is what right looks like, and here is what almost-right looks like and why it's still wrong."

**Missing Teacher 1: Contrastive Learning (MCQ Library)**

In v3.05, we built an MCQ (Multiple Choice Question) system. A misclassified case became a structured lesson:
> *Q: Does this sentence describe an ADE? "Patient developed rash after amoxicillin."*
> *A) YES — explicit temporal relation, named drug, documented adverse outcome ← CORRECT*
> *B) NO — rash could be coincidental ← WRONG: temporal proximity after drug start meets Bradford Hill criteria*
> *C) UNCERTAIN — need rechallenge data ← WRONG: single episode sufficient for ADE classification*

We dropped MCQs in v1 to avoid FM-3 (MCQ complexity harm). But FM-3 was caused by giving MCQs to immature columns regardless of BCM state. The fix is BCM-gated MCQ depth, not MCQ removal.

**Wrong answers teach as much as right ones.** This is the contrastive signal every supervised learning system needs.

**Missing Teacher 2: Rejected Proposal Memory**

The LLM proposed `has_toxicity + negations` as a genesis trigger in R7, R9, and effectively R10. Three times. With no memory that it tried before. A student who fails the same exam question three times without reviewing the answer is not learning — they're repeating.

Every failed genesis attempt encodes information: this trigger pattern, on this corpus, at this developmental stage, did not help global performance. That information must be fed back to the LLM before its next genesis proposal.

**Missing Teacher 3: Adult Guidance (Meta-Agent)**

When a child's fever doesn't break after 3 days, you call the doctor. The doctor doesn't just take the temperature — they examine the full picture, make a diagnosis, and prescribe an intervention.

Our cortex had no doctor. When F1 declined for 3 consecutive rounds (R7-R8-R9), nothing examined the full picture. The system just kept running the same protocol. A meta-agent LLM call — given the full cortex state, F1 trajectory, error patterns, and rejected proposal history — could diagnose: *"The precision problem is driven by ROOT handling too many borderline cases. COL_MEDICAL_DISCUSSION would relieve this if the trigger were [X]. Previous attempts failed because [Y]."*

The LLM is capable of this reasoning. We were not asking it to do this.

---

## Entry 5 — June 26, 2026: Toward v2.0-cortex

**Design principle: The LLM is the Adult. The Cortex is the Child.**

In v1, the LLM was a tool — call it, get an answer, move on. It had no awareness of the system it was embedded in. No memory of its own successes and failures. No ability to look at the developing cortex and say "this child needs a different kind of help right now."

In v2, the LLM operates in three distinct developmental roles:

**Role 1 — Specialist Classifier** (unchanged)
Classify individual cases using RAG context, MemoryTraces, and MCQs.

**Role 2 — Structured Teacher** (new)
After each round, convert error cases into MCQ lessons. BCM-gated: immature columns (LTD) receive simpler lessons. Mature columns (LTP) receive full contrastive MCQs with multiple distractors and detailed rationale.

**Role 3 — Diagnostic Physician** (new)
When F1 declines or the cortex is stagnating, the meta-agent receives a complete status report and returns a diagnosis with specific interventions. This is the adult looking at the child's development and saying: "Something is wrong here. Here is what I think it is. Here is what we should try."

**New mechanisms in v2:**

*Shadow Column Period:* New columns observe routing for 1 round before activating. They warm up on real cases before their first live classification. Prevents the "cold column" problem where an untrained specialist immediately handles cases it has never processed.

*Trigger-Scoped Genesis Probe:* Instead of evaluating a new column's impact on all 50 probe cases, evaluate only on cases that match the column's trigger. A column designed for 10% of cases should be judged on that 10%, not on the 90% it was never meant to touch.

*Rejected Proposal Memory:* A persistent log, growing across all rounds, of every failed genesis attempt. The LLM reads this before every new proposal. It cannot propose the same trigger pattern twice.

**What success looks like for v2:**
- Genesis survival rate > 50% (v1: 20%, and both survivals were early-phase luck)
- Specialists earning LTP before R10 (v1: COL_SHORT earned LTP at R14, COL_NEGATION at R17)
- F1 monotonically improving or stable through DEVELOPMENTAL phase (v1: peaked R3, dipped R7-8)
- Meta-agent interventions measurably changing trajectory

---

## The Overarching Question

We are building, version by version, toward something that has not been demonstrated cleanly in the literature: **a system that learns to learn — not by gradient descent on fixed parameters, but by dynamically building cognitive structure around a reasoning core.**

The parallel to developmental neuroscience is deliberate and testable. The BCM equation is not a metaphor — it is the actual update rule, implemented in Python, running on clinical text. The critical period threshold is not inspiration — it is a sigmoid function derived from Hensch's empirical data.

If we can show that:
1. A biologically-grounded learning architecture outperforms both a static LLM and a naive self-growing tree
2. The failure modes are interpretable through neuroscience (not just debugging)
3. The interventions (MCQ pedagogy, meta-agent guidance) measurably improve developmental trajectory

...then we have a contribution to both clinical NLP and computational neuroscience.

**The ultimate goal:**
A general-purpose computational neuroscience learning agent. Not tuned to ADE classification. Not tuned to any fixed task. A system architecture that can be instantiated with different seed columns and different task configs to learn any structured classification problem — guided by the same developmental biology that shaped the human neocortex over 500 million years of evolution.

---

## Findings Log

| Date | Finding | Significance | Version |
|------|---------|--------------|---------|
| 2026-06-23 | Single LLM call = flat performance | Baseline established | v0.1 |
| 2026-06-23 | RAG context improves recall sharply | Hippocampal retrieval is real | v1.0 |
| 2026-06-24 | Parallel routes improve recall + precision | Cortical streams work | v1.0 |
| 2026-06-25 | v3.04 peaks at seed config, collapses with growth | FM-1: FIFO routing dilution | v3.04 |
| 2026-06-25 | Routing dilution is FM-1: specificity-based routing fixes it | Winner-take-all biology | v3.04 |
| 2026-06-26 | BCM dynamics fire correctly; homeostasis prevents cliff | Core biology works | v1.0-cortex |
| 2026-06-26 | 8/10 genesis rollbacks in DEVELOPMENTAL phase | Probe scoping problem + no MCQ | v1.0-cortex |
| 2026-06-26 | LLM repeats same genesis pattern 3× without memory | Rejected proposal memory needed | v1.0-cortex |
| 2026-06-26 | Net learning over 20 rounds ≈ 0 (R1=0.778, R20=0.759) | Missing: pedagogy + adult guidance | v1.0-cortex |
| 2026-06-26 | COL_MEDICAL_DISCUSSION (spec=3, Jaccard=0.073) survived; spec=1 proposals all failed | High specificity + low overlap = genesis success | v1.0-cortex |
| 2026-06-26 | First apoptosis: COL_TEMPORAL_OBSERVATION starved by higher-spec sibling | Winner-take-all can starve legitimate specialists | v1.0-cortex |
| 2026-06-26 | v2.0-cortex designed and built: MCQLibrary + RejectedProposalMemory + MetaAgent | Three missing teachers now implemented | v2.0-cortex |
| 2026-06-26 | v2.0-cortex R10 F1=0.844 — MetaAgent JSON parse failures at R6 and R10 | Markdown in LLM JSON responses breaks json.loads; needs repair fallback | v2.0-cortex |
| 2026-06-26 | MCQ Learner plateau at F1=0.80–0.83 across 16 rounds | Root: coarse boolean features contradicted each other; threshold oscillation 0.30↔0.80; lesson retirement by global not pattern F1 | MCQ Learner |
| 2026-06-27 | Apex Learner designed: δ-RPE + contrastive pairs + Gamma-Theta + EMA + ACh + 3-timescale memory | Synthesizes all failures + neuroscience theory into next generation | v-Apex |

---

## Entry 6 — June 26, 2026: Building v2.0-cortex

**What we built:**

nexus_cortex_v2.py is a complete replacement for v1 that preserves all working biological mechanisms and adds the three missing learning components identified from v1's 20-round analysis.

**The five new mechanisms, each with a biological rationale:**

**1. MCQLibrary — Contrastive Lessons**

The old WorkingMemory was an error buffer. It told the column: "you got these cases wrong." But knowing you got something wrong without seeing why the right answer is right — and why wrong answers are wrong — produces almost no learning signal. This is why v1 columns stayed in LTD through 20 rounds despite accumulating hundreds of error cases.

MCQLibrary generates a structured lesson from each error: the correct answer with rationale, plus 2–3 wrong answers with explicit explanations of why each is wrong. The LLM generates these at a cost of 1 call per MCQ. BCM gates the depth — LTD columns (rehearsal_weight≈0.30) receive 2 MCQs, LTP columns receive up to 8. The contrastive signal is qualitatively different from raw errors.

VanLehn (2011): "Students who study worked examples with correct and incorrect solutions learn more than those who study only correct solutions."

**2. RejectedProposalMemory — Genesis Failure Log**

In v1, the LLM proposed `has_toxicity + negations` (effectively) in R7, R9, and R10. No memory of prior failures. RejectedProposalMemory maintains a growing log of every failed genesis: trigger, spec, Jaccard, probe Δ, estimated reason.

Before every genesis LLM call in v2, the full rejection history is provided. The LLM reads: "I tried this pattern in rounds 7, 9, 10 — each time probe Δ was negative — I need to think differently." The same contrastive signal that MCQs provide for classification, applied to genesis.

**3. MetaAgent — Diagnostic Physician**

When F1 declines for 2 consecutive rounds, a meta-agent LLM call fires. The LLM receives the full cortex state, F1 trajectory, rejected proposal history, and a sample of this round's error cases. It returns a structured diagnosis: root cause + up to 3 interventions (prompt refinement guidance, column pruning recommendations, genesis proposals grounded in error analysis, threshold direction).

The MetaAgent is the attending physician who sees the developing brain not case-by-case but as a whole, who can say: "The FP problem is ROOT-driven. The threshold needs to come up. The pattern in these errors suggests a column covering has_report + has_short has not been tried."

**4. Shadow Column Period**

New columns enter shadow_mode after genesis. For 1 round, they observe which cases would have been routed to them — but do not intercept any. They build up shadow_cases. After the shadow round, a trigger-scoped probe is run. Only then do they either activate fully or get rolled back.

This prevents the "cold column" problem: v1 probed columns immediately after genesis, before any rehearsal. A column with 0 MCQs, 0 MemoryTraces, and 0 warm-up was being judged against a ROOT with 10 MemoryTraces and 60 error cases in its buffer. It was never a fair test.

**5. Trigger-Scoped Genesis Probe**

The v1 genesis probe evaluated new columns on 50 random cases from the probe pool. If a column's trigger fired on 10% of the corpus, the 90% of probe cases it would never touch still counted against it. A column that was excellent on its 10% but irrelevant to the other 90% would show a flat or negative aggregate delta.

The v2 scoped probe evaluates only cases matching the column's trigger, comparing column F1 vs ROOT F1 on those same cases. If fewer than 10 trigger-matched cases exist, it falls back to global probe. This is the right comparison: does this column outperform ROOT on the cases it was designed to handle?

**What we expect from v2:**

Genesis survival rate above 50%. Columns earning LTP before R10 because MCQs are providing the contrastive learning signal they need to discriminate. F1 trajectory that does not peak at R3 because the meta-agent catches early declines and intervenes. No repeated genesis patterns because the rejection memory enforces novelty.

If v2 shows these patterns, it confirms that the v1 stagnation was not a fundamental limit of the biologically-grounded architecture — it was three missing pedagogical mechanisms. The biology was correct. The teaching was absent.

---

*This diary continues with each run. The findings log grows. The system grows. The understanding grows.*

*The goal is not to build a better classifier. The goal is to build a system that teaches us something we did not know about how learning works.*

---

## Entry 7 — June 26, 2026: v2.0-cortex Run — Three Teachers, One Problem

**What ran:**
nexus_cortex_v2.py — full implementation of MCQLibrary, RejectedProposalMemory, MetaAgent, Shadow Column Period, and Trigger-Scoped Genesis Probe.

**Results by round:**
| Round | Eval F1 | Notes |
|-------|---------|-------|
| 1     | 0.788   | Blank slate, no lessons injected |
| 2     | 0.810   | MCQs beginning to accumulate |
| 3     | 0.822   | Steady improvement — contrastive signal working |
| 4     | 0.831   | Threshold stabilizing |
| 5     | 0.836   | Near-miss pattern visible |
| 6     | MetaAgent FAIL | JSON parse error: markdown in LLM JSON response |
| 7     | 0.829   | Slight regression after MetaAgent failure |
| 8     | 0.835   | Recovery |
| 9     | 0.840   | Best so far |
| 10    | 0.844   | **Peak** — then MetaAgent parse failure again |

**The JSON parse failure:**
The MetaAgent received a complex prompt with full cortex state and returned formatted JSON with embedded newlines inside string values. `json.loads()` failed with `"Expecting ',' delimiter: line 15 column 6 (char 1856)"`. The LLM was producing valid-looking but technically invalid JSON — newlines inside strings, trailing commas, markdown fences.

This is a solvable engineering problem: regex extraction + json-repair library + simpler output schema. But it interrupted two consecutive meta-agent interventions exactly when the system needed them most (R6–10 is the prime developmental window where the meta-agent should be most active).

**What worked in v2 vs v1:**
- MCQLibrary delivered measurable benefit: columns reached LTP by R7 vs R14 in v1
- RejectedProposalMemory eliminated repeated genesis proposals — no pattern was attempted twice
- Shadow column period improved genesis survival rate from 20% → 40%
- Trigger-scoped probe correctly identified genuinely useful columns

**What still didn't work:**
- Peak at R10 (0.844) with slow growth afterward — the lessons were helping but the rate of gain was insufficient
- Column proliferation began at R8: too many active columns fighting for routing priority
- MCQ lesson retirement logic was global F1 based, not pattern-level — useful lessons were retired when global F1 dipped

**The MCQ Learner parallel run:**
Simultaneously tested a simpler approach: a flat MCQ-based learner with no columnar routing, operating on 1000-case batches over 16 rounds.

Results: peak F1 = 0.8276 at R2, plateau at ~0.80 through R16. Three failure modes identified:
1. **Feature-level contradiction**: `has_drug_name` appeared in both FP and FN patterns, causing MCQ lessons to contradict each other. A single boolean feature cannot capture the semantics needed for a discriminating lesson.
2. **Threshold oscillation**: Fresh calibration sweep each round swung threshold 0.30↔0.80. The system was chasing its own signal.
3. **Lesson retirement at wrong level**: Global F1 was the retirement trigger. Pattern-level absorption was never measured.

**The convergent diagnosis across both v2 and MCQ Learner:**
Both systems learned — but not fast enough, not cleanly enough, and not in a way that transferred to future cases. The missing mechanism is not more structure. It is better signal at the teaching unit level.

A good teacher does not show you the average of all your mistakes. A good teacher finds the case where you almost got it right and shows you exactly where your reasoning diverged from the correct path.

---

## Entry 8 — June 27, 2026: The Computational Neuroscience Perspective

**A thought experiment: 40 years of neuroscience, applied**

If a computational neuroscientist who built some of the earliest neural networks — who then spent 40 years on memory and learning — looked at NEXUS at this moment, what would they say?

They would recognize the architecture. BCM theory: correct. Critical period: correct. Homeostasis: correct. Engrams: correct. But they would identify what is missing before they said anything else.

**"You have the Hebbian machinery but not the Hebbian insight."**

Hebb's rule says: neurons that fire together, wire together. But the complement — neurons that fire apart, wire apart — is equally important. Your system logs errors. It does not systematically find the correct case nearest to each error and learn the distinction between them. That distinction — the boundary, not the center — is where all learning happens.

The sensory cortex does not learn "what a face looks like" by averaging all faces. It learns to distinguish one face from a similar face. The discriminative boundary is the engram. Your error log builds a catalog of the interior, not the boundary.

**"Your teacher is talking at the student, not with them."**

LTP requires pre- and post-synaptic co-activation within a narrow time window (Magee & Johnston 1997). You give the LLM an error case. The LLM classifies. There is no co-activation — the LLM's own prediction error is not paired with the correct answer within the same forward pass. The lesson arrives in the next round's context, cold, with no reference to the specific reasoning error.

Contrastive pairs solve this: "You said NOT_ADE for this sentence because you saw a therapeutic framing. Here is the nearest sentence that genuinely IS NOT_ADE because of the same therapeutic framing — and here is what separates the two." Now the LLM's error and the correct boundary are co-activated in the same context window.

**"You're measuring confidence but not using it as a learning signal."**

Dopamine neurons encode reward prediction error: δ = reward − predicted_reward. The largest learning signal is not the reward or the punishment — it is the *surprise*. A confident wrong answer is more surprising than an uncertain wrong answer. Your system logs errors equally regardless of confidence.

Weight errors by confidence × is_wrong. A confident false positive (the model was sure it was ADE, it was NOT_ADE) teaches far more than an uncertain false positive. The case the model was certain about and wrong about — that is the crack in the understanding, not just a random mistake.

**"You need sleep."**

The hippocampus replays experiences during slow-wave sleep. The neocortex slowly consolidates. The next morning, you wake with integrated understanding, not just a list of yesterday's events.

Your system processes errors within the same round they occur. There is no consolidation phase where accumulated lessons are synthesized into a coherent causal model. After 10 rounds, the LLM context contains a growing list of specific lessons, not a rewritten understanding of the domain.

Every 3 rounds, consolidate: give the LLM all accumulated lessons and ask it to synthesize a 2-3 sentence causal model. This becomes the new system prompt — not a list of lessons, but a new understanding. The system stops consulting notes and starts thinking from the integrated knowledge.

**The prescription:**
Replace MCQs with contrastive pairs (boundary teaching). Weight errors by δ = confidence × is_wrong (dopamine RPE). Consolidate lessons into a causal model every 3 rounds (sleep phase). Keep near-miss cases as boundary exemplars (LTP pairing requirement). Run a second pass on the hardest cases with fresh lessons (theta replay).

This is not a list of improvements. It is a single coherent theory of how learning works, applied to a machine learning system.

---

## Entry 9 — June 27, 2026: NEXUS Apex — The Next Generation

**Design principle: From blank slate to taught adult**

Every prior version had some form of lesson injection from round 1. We were feeding the system answers before it had built up enough error history to know what it needed to learn. The system had no sense of its own ignorance.

In Apex, rounds 1–3 are a blank slate. Pure classification. No lessons injected. The system builds an error history — discovering where it fails, how confidently it fails, and what the structure of its failures is. Only from R4 onward do lessons begin to appear, grounded in a real understanding of what the system gets wrong.

This mirrors developmental biology precisely: the embryonic critical period allows exploration before the formation of strong synaptic preferences. The brain does not start with adult synaptic weights. It starts with broad potential and narrows through experience.

**The twelve mechanisms of Apex:**

1. **δ-weighted prediction error** (Schultz 1997): `δ = confidence × is_wrong`. Errors sorted by δ. Only high-δ errors become contrastive pairs.

2. **Rationale-based error taxonomy** (zero LLM cost): The LLM's own rationale is parsed for the type of error — temporal_confusion, therapeutic_goal, report_context, negation_confusion, causal_ambiguity, completeness_confusion. Errors grouped by type before pair generation.

3. **Direction verification before pair generation**: A pattern must show ≥70% same-direction errors (FP or FN) before a directional lesson is generated. Mixed-direction patterns get a contrastive pair but no directional lesson — the system does not overfit to a noisy signal.

4. **Contrastive pair generation** (1 LLM call per pair): anchor error + nearest opposite-label case from RAG → LLM generates lesson + key distinction. The teaching unit is the boundary, not the error.

5. **Embedding-based lesson retrieval**: During classification, lessons are retrieved by cosine similarity to the current case embedding — not by boolean feature overlap. The lesson about "temporal following causing FP" surfaces when a temporally-framed case is encountered, not when `has_following=True`.

6. **Near-miss mining** (Bliss & Lømo 1973 — LTP pairing): Correct predictions with confidence < 0.62 are logged as near-misses. They sit on the decision boundary. They get a second look in theta pass and become boundary examples in future rounds.

7. **Two-pass Gamma-Theta architecture** (Lisman & Jensen 2013): Gamma pass classifies all 1000 cases. Theta pass reclassifies the top-50 δ errors + top-50 near-misses with fresh lessons included. Theta correction rate is the absorption signal.

8. **EMA threshold stability**: `threshold = 0.70 × prev + 0.30 × calibrated`. Only updates if Δ > 0.08. Eliminates the 0.30↔0.80 oscillation.

9. **ACh plasticity gating** (Hasselmo 1999): `plasticity = tanh(error_rate × 3)`. The number of pairs generated per round scales with plasticity. High error rate → high plasticity → many new lessons. Low error rate → the system has learned — stop generating aggressively.

10. **Three-timescale memory** (McClelland 1995): Fast (per-case δ log), Medium (per-round contrastive pairs), Slow (per-3-rounds causal model synthesis). Each timescale accumulates at a different rate and serves a different function.

11. **Consolidation every 3 rounds**: LLM synthesizes all accumulated lessons into a causal model. This replaces the system prompt — not a note appended, but a new understanding that the model starts from. The system stops consulting the lesson list and starts speaking from the integrated understanding.

12. **Absorption tracking**: Theta correction rate measures whether today's lessons worked on today's hardest cases. Pairs that have been in the active set for 3+ rounds and are still generating theta corrections are promoted to CORE (never retired). Pairs that do not generate theta corrections are gradually retired.

**What is intentionally absent:**
Columnar routing. In every version from v1.0-cortex onward, the routing architecture added complexity before the learning mechanism was validated. Apex tests the learning hypothesis first. If contrastive pairs + EMA threshold + three-timescale consolidation can drive F1 above 0.90 with a flat single-prompt classifier, we will have proved the learning mechanism works. Then we add routing in v4.

**Target metrics:**
- F1 > 0.85 at R4 (beginning of lesson injection phase)
- F1 > 0.90 by R8–10
- Theta correction rate > 25% (lessons absorbing on hardest cases)
- No threshold oscillation (EMA stability)
- Causal model synthesis producing coherent 2–3 sentence clinical understanding by R6

**Architecture files:**
- `nexus_db_apex.py` — SQLite three-timescale memory (errors, near_misses, contrastive_pairs, structural_knowledge, round_stats)
- `nexus_apex.py` — Full Apex Learner main loop
- `run_apex.sh` — Run script (fresh/warm/N-round/mock modes)

---

## Version Summary Table

| Version | Architecture | Peak F1 | When | Failure Mode |
|---------|-------------|---------|------|--------------|
| v0 | Single LLM call | ~0.70 | R1 | No learning |
| v1.0 RAG+routes | Parallel expert agents + RAG | ~0.78 | R1 | No cross-round memory |
| v3.04 tree | Self-growing FIFO tree | **0.9412** | R1 | FM-1: FIFO dilution → collapse to 0.73 by R20 |
| v3.05 tree+MCQ | v3.04 + MCQ on mature nodes | 0.82 | R3 | FM-3: MCQ harm on immature nodes |
| v1.0-cortex | BCM + competitive routing + 6 bio mechanisms | 0.8182 | R3 | No pedagogy — net zero learning R1→R20 |
| v2.0-cortex | v1 + MCQLibrary + MetaAgent + RejectedProposalMemory | 0.844 | R10 | MetaAgent JSON parse failures; column proliferation |
| MCQ Learner | Flat learner, boolean MCQs, 1000-case batches | 0.8276 | R2 | Feature contradiction; threshold oscillation; wrong retirement level |
| **v-Apex** | δ-RPE + contrastive pairs + Gamma-Theta + EMA + ACh + 3-timescale | **TBD** | **TBD** | Baseline not yet established |

---

*This diary continues with each run. The findings log grows. The system grows. The understanding grows.*

*The goal is not to build a better classifier. The goal is to build a system that teaches us something we did not know about how learning works.*
