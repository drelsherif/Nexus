# NEXUS — What It Is and How It Works
### A plain-language explanation for anyone

---

## The problem it solves

Every day, medical researchers publish thousands of papers describing patients who had bad reactions to drugs. Someone needs to read those sentences and decide: *"Is this an adverse drug event — a real harmful reaction — or something else?"*

Doing this by hand takes expert humans and thousands of hours. Current AI tools can do it, but they are frozen — they learn once from a large labeled dataset, then stop. They cannot improve from new cases, cannot explain their reasoning, and cannot adapt when the literature evolves.

NEXUS is built to do better.

---

## The analogy: a medical resident

Think of NEXUS as a medical resident on their first day.

On day one, they know the basics — what an adverse drug event is, what negation means, what causal language looks like. They can handle most cases, but they make mistakes.

Over time, the resident sees more patients. When they keep getting a certain type of case wrong, they take note. They read about it. They form a rule in their head: *"Sentences that start with 'We report a case of...' are introductory — don't assume they describe the actual event."* They remember the specific hard cases that taught them that rule.

They also start to recognize subspecialties. One type of case — sentences with negation — gets routed to the "negation thinking module" in their brain. Another type — very short telegraphic notes — gets handled differently. Over time, their thinking becomes more organized and more accurate.

That is NEXUS.

---

## How it works, step by step

**1. The tree (the resident's thinking structure)**
NEXUS organizes its reasoning as a decision tree. Each node in the tree is a specialist — a focused classifier that handles one type of sentence. When a new sentence arrives, NEXUS routes it to the right specialist based on what the sentence contains. Does it have negation words? Go to the negation node. Is it a short telegraphic sentence? Go to the short-sentence node. Everything else goes to the general node (ROOT).

**2. The memory systems (how it learns)**
Each node has three types of memory, just like a doctor:
- **Case examples (RAG):** When classifying a new sentence, the node retrieves the most similar sentences it has ever seen and uses those as context — like a doctor recalling "I've seen something like this before."
- **Hard case flashcards (MCQs):** When the node gets a case wrong, it creates a flashcard — the sentence, the wrong answer, the right answer, and why. It reviews these flashcards every time similar cases arrive.
- **General principles:** When enough similar errors cluster together, the node distills them into a written principle — a lasting rule that changes how it reasons going forward.

**3. The growth mechanism (the "aha moment")**
When a cluster of errors grows large enough — many similar cases that the node keeps getting wrong — NEXUS fires what it calls a Sharp-Wave Ripple (SWR) event, borrowed from neuroscience. This is the system's "aha moment." It asks: *"Should I create a new specialist node for this pattern, or is a new principle enough?"*

If a new specialist would improve performance, NEXUS grafts a new branch onto the tree. If not, it writes a principle and moves on.

**4. What it does NOT do**
NEXUS does not retrain its neural network. It does not require a GPU. It does not need thousands of labeled examples before it becomes useful. It starts from four basic nodes and grows from there, using only the cases it actually encounters.

---

## What makes it different from standard AI

| Standard AI model (e.g. PubMedBERT) | NEXUS |
|---|---|
| Trained once, then frozen | Learns continuously from every case |
| Black box — no explanation | Every decision is traceable to a principle or example |
| Requires thousands of labeled examples | Starts from 4 seed nodes, grows from experience |
| Needs GPU to retrain | Runs on any computer, improves with API calls only |
| Treats all cases the same | Routes cases to specialists based on their features |

---

## What it has achieved so far

In early testing on a standard medical literature dataset (ADE-Corpus-V2), NEXUS:
- Started at F1=0.80 with four basic seed nodes
- Grew to F1=0.93 over 12 rounds of 200 cases each
- Reduced redundant AI calls by 91% through intelligent deduplication
- Recovered from disruptive learning events (new node grafts) through self-correction
- Did all of this with no GPU, no manual retraining, and full auditability

---

## Why it matters for medicine

Medicine generates more text than any human or static AI can read. The patterns in that text change as new drugs emerge, new side effects are discovered, and new clinical language evolves. A system that learns continuously, reasons transparently, and grows its expertise from experience is not just a faster version of what exists — it is a fundamentally different approach to how AI can be used in clinical research.

NEXUS is a prototype of that approach.

---

*Designed and built by Yasir El-Sherif, MD — Northwell Health*
