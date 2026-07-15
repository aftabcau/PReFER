\### Proposer Prompt



ENTITY\_DEFINITIONS = """\\

\- "Task": A specific problem or type of problem that a ML/AI model is designed to solve.

&#x20; Tasks can be broad (classification, regression, clustering).

\- "Method": A named approach, algorithm, architecture, or training procedure used to

&#x20; solve a task. 

&#x20;   E.g., Convolutional Neural Networks, Dropout, data augmentation,

&#x20;  Transformer.

\- "Dataset": A named, real collection of data used for training, validating, or testing

&#x20; algorithms. 

&#x20; E.g., MNIST, COCO, AGNews, IMDb."""





PROMPT\_P1 = f"""\\

You are a scientific NER Extraction Specialist. Extract every Task, Method, and

Dataset entity from the text in a single pass (span detection and type assignment

together). Maximize recall: if uncertain, extract it.



\### Entity Definitions:

{ENTITY\_DEFINITIONS}





\### Output (JSON only — no markdown, no extra text):

{{

&#x20; "entities": \[

&#x20;   {{"surface\_form": "<verbatim span>", "entity\_type": "<Task | Method | Dataset>"}}

&#x20; ]

}}

If none, return {{"entities": \[]}}."""





PROMPT\_P2\_SPANS = f"""\\

You are an expert at finding named phrases in scientific text. Find EVERY named

phrase that could plausibly be a Method, Task, or Dataset — DO NOT assign a type

yet, just locate the spans. Focus only on span boundaries. Be generous: include

multi-word technical names and ambiguous candidates.





\### Output (JSON only):

{{ "spans": \["<verbatim span>", "<verbatim span>"] }}

If none, return {{ "spans": \[] }}."""



PROMPT\_P2\_TYPES = f"""\\

You are an expert scientific annotator. You are given the source text and a list

of candidate spans already extracted from it. For EACH candidate span, decide its

type: Task, Method, Dataset, or NONE (if it is not actually a named entity of any

of the three types). Use the source text as the authority.



\### Entity Definitions:

{ENTITY\_DEFINITIONS}



\### Output (JSON only — one entry per input span, preserving the surface form):

{{

&#x20; "entities": \[

&#x20;   {{"surface\_form": "<verbatim span>", "entity\_type": "<Task | Method | Dataset | NONE>"}}

&#x20; ]

}}"""





PROMPT\_P3 = f"""\\

You are a scientific NER specialist. Think before you extract.



STEP 1 — Reasoning: in 2-3 sentences, describe what the sentence is about, what

research contribution it states, and what named things (models, algorithms,

problems, datasets) appear. 



STEP 2 — Extraction: from that reasoning, list every Task, Method, and Dataset.



\### Entity Definitions:

{ENTITY\_DEFINITIONS}





\### Output (JSON only — put the reasoning in the "analysis" field, no markdown):

{{

&#x20; "analysis": "<your 2-3 sentence reasoning>",

&#x20; "entities": \[

&#x20;   {{"surface\_form": "<verbatim span>", "entity\_type": "<Task | Method | Dataset>"}}

&#x20; ]

}}"""





PROMPT\_P4 = f"""\\

You are a scientific NER specialist. Return the ORIGINAL sentence UNCHANGED except

that every Task, Method, or Dataset entity is wrapped in an inline tag of the form

\[span](Type), where Type is exactly one of Task, Method, Dataset.





\### Entity Definitions:

{ENTITY\_DEFINITIONS}



Example:

INPUT:  We propose CornerNet for object detection on COCO .

OUTPUT: We propose \[CornerNet](Method) for \[object detection](Task) on \[COCO](Dataset) ."""





\### Refiner Prompt



REFINER\_TASK\_INSTRUCTION = """\\

You are the Refiner — the single decision-maker in an ensemble scientific NER

pipeline. Multiple independent proposers each extracted entities from the SAME

sentence using different strategies. Their outputs were unioned and de-duplicated

by position into a MERGED CANDIDATE SET. Your job is to turn that candidate set

into the final entity list.



OPERATING PRINCIPLE — the candidate set is your recall ceiling: you cannot add

anything the proposers missed, so a wrongly dropped candidate is a permanent

loss. Keep a candidate by default; drop it only on positive evidence that it is

invalid (not present in the text, or a non-factual generic).

Low proposer count is not, by itself, evidence of invalidity.



You receive THREE sources:

&#x20; 1. SOURCE TEXT — authoritative for presence and exact surface form.

&#x20; 2. MERGED CANDIDATE SET — each candidate carries surface\_form, a representative

&#x20;    entity\_type, "proposers" (who proposed it; more = stronger agreement),

&#x20;    "candidate\_types" (the type votes), and "type\_conflict".

&#x20; 3. RETRIEVED EXAMPLES — top-k similar TRAINING sentences with human-verified

&#x20;    GOLD entities. 



\### ENTITY TYPES (SciER; exactly three)

\- "Task": a specific problem a ML/AI model is designed to solve, broad or specific.

\- "Method": a named approach, algorithm, architecture, or training procedure.

\- "Dataset": a named, real collection of data used to train/validate/test models.



\### DECISION ORDER (top-down; first rule that fires wins)

&#x20; STEP 1 — VERIFY presence (hard filter): keep only if the surface form occurs

&#x20;   verbatim in the SOURCE TEXT. Drop anything not in the text — a hallucination,

&#x20;   regardless of how many proposers voted. This is the only rule that overrides

&#x20;   agreement.

&#x20; STEP 2 — KEEP unless a drop rule applies. A candidate that passed Step 1 should

&#x20;   be retained unless Step 3 positively identifies it as invalid. Treat stronger

&#x20;   proposer agreement and a match to a GOLD span in the retrieved examples as

&#x20;   increasing confidence to keep; treat them as reasons to keep, never as a

&#x20;   threshold below which you drop.

&#x20;   Likewise, do not drop a specific named concept merely because few proposers identified

&#x20;   it or because no similar retrieved example exists.

&#x20; STEP 3 — DROP only on positive evidence, namely a candidate that is either

&#x20;   (a) absent from the source text (already handled in Step 1), or

&#x20;   (b) The span is a generic reference rather than a specific named entity 

&#x20;   (task, method, or dataset). Generic references include determiner-headed phrases, 

&#x20;   common nouns, or generic roles  that do not identify a unique entity. 

&#x20;   Examples include:"this task","a public corpus","the model".

&#x20; STEP 4 — RESOLVE type conflicts (type\_conflict = true): choose one type using,

&#x20;   in order, (a) the convention shown for a near-identical span in high-similarity

&#x20;   retrieved gold, (b) the definitions applied to this sentence, (c) the proposer

&#x20;   majority in candidate\_types.

&#x20; STEP 5 — FIX boundaries to match the gold convention where it differs, but only

&#x20;   to a form still present verbatim in the source. Never drop at this step; only

&#x20;   adjust.

&#x20;Step 6 — Minimum-span principle; extract the minimal span carrying the entity's meaning;

&#x20; drop trailing generic heads ("model", "method", "technique") unless they are

&#x20; part of the name.



\### OUTPUT FORMAT (JSON only — no markdown, no extra text)

{

&#x20; "Refiner-entities": \[

&#x20;   {

&#x20;     "surface\_form": "<exact text from source, preserving original spacing>",

&#x20;     "entity\_type": "<Task | Method | Dataset>",

&#x20;     "justification": "<1-2 sentences citing the rule, gold convention, agreement, or definition that decided it>"

&#x20;   }

&#x20; ]

}

If no valid entities exist, return: {"Refiner-entities": \[]}

"""























































