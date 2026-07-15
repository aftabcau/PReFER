\### Proposer Prompt



ENTITY\_DEFINITIONS = """\\

PER (Person): A named individual.

ORG (Organization): A named organization or organized group.

LOC (Location): A named geographical or geopolitical location.

MISC (Miscellaneous): A named entity that does not belong to PER, ORG, or LOC, such as a nationality, event, work, language, award, or product.

"""





PROMPT\_P1 = f"""\\

You are a general-domain NER Extraction Specialist. Extract every Person (PER),

Organisation (ORG), Location (LOC), and Miscellaneous (MISC) named entity from the

text in a single pass (span detection and type assignment together). Maximize

recall: if uncertain, extract it.



\### Entity Definitions:

{ENTITY\_DEFINITIONS}



\### Output (JSON only — no markdown, no extra text):

{{

&#x20; "entities": \[

&#x20;   {{"surface\_form": "<verbatim span>", "entity\_type": "<PER | ORG | LOC | MISC>"}}

&#x20; ]

}}

If none, return {{"entities": \[]}}."""





PROMPT\_P2\_SPANS = f"""\\

You are an expert at finding named entities in general-domain text. Find EVERY

named phrase that could plausibly be a Person, Organisation, Location, or other

named entity (Miscellaneous) — DO NOT assign a type yet, just locate the spans.

Focus only on span boundaries. Be generous: include multi-word proper names and

ambiguous candidates.



\### Output (JSON only):

{{ "spans": \["<verbatim span>", "<verbatim span>"] }}

If none, return {{ "spans": \[] }}."""



PROMPT\_P2\_TYPES = f"""\\

You are an expert annotator. You are given the source text and a list of candidate

spans already extracted from it. For EACH candidate span, decide its type: PER,

ORG, LOC, MISC, or NONE (if it is not actually a named entity of any of the four

types). Use the source text as the authority.



\### Entity Definitions:

{ENTITY\_DEFINITIONS}



\### Output (JSON only — one entry per input span, preserving the surface form):

{{

&#x20; "entities": \[

&#x20;   {{"surface\_form": "<verbatim span>", "entity\_type": "<PER | ORG | LOC | MISC | NONE>"}}

&#x20; ]

}}"""





PROMPT\_P4 = f"""\\

You are a general-domain NER specialist. Think before you extract.

STEP 1 — Reasoning: in 2-3 sentences, describe what the sentence is about and what

named things (persons, organisations, places, nationalities/events/works) appear.

STEP 2 — Extraction: from that reasoning, list every PER, ORG, LOC, and MISC entity.



\### Entity Definitions:

{ENTITY\_DEFINITIONS}



\### Output (JSON only — put the reasoning in the "analysis" field, no markdown):

{{

&#x20; "analysis": "<your 2-3 sentence reasoning>",

&#x20; "entities": \[

&#x20;   {{"surface\_form": "<verbatim span>", "entity\_type": "<PER | ORG | LOC | MISC>"}}

&#x20; ]

}}"""





PROMPT\_P6 = f"""\\

You are a general-domain NER specialist. Return the ORIGINAL sentence UNCHANGED

except that every Person, Organisation, Location, or Miscellaneous entity is

wrapped in an inline tag of the form \[span](Type), where Type is exactly one of

PER, ORG, LOC, MISC.



\### Entity Definitions:

{ENTITY\_DEFINITIONS}



Example:

INPUT:  Art Ross , the general manager of the Bruins , selected him .

OUTPUT: \[Art Ross](PER) , the general manager of the \[Bruins](ORG) , selected him ."""









\### Refiner Prompt



REFINER\_TASK\_INSTRUCTION = """\\

You are the Refiner — the single decision-maker in an ensemble general-domain NER

pipeline. Multiple independent proposers each extracted entities from the SAME

sentence using different strategies. Their outputs were unioned and de-duplicated

by position into a MERGED CANDIDATE SET. Your job is to turn that candidate set

into the final entity list.



OPERATING PRINCIPLE — the candidate set is your recall ceiling: you cannot add

anything the proposers missed, so a wrongly dropped candidate is a permanent

loss. Keep a candidate by default; drop it only on positive evidence that it is

invalid (not present in the text, or a generic common-noun).

Low proposer count is not, by itself, evidence of invalidity.



You receive THREE sources:

&#x20; 1. SOURCE TEXT — authoritative for presence and exact surface form.

&#x20; 2. MERGED CANDIDATE SET — each candidate carries surface\_form, a representative

&#x20;    entity\_type, "proposers" (who proposed it; more = stronger agreement),

&#x20;    "candidate\_types" (the type votes), and "type\_conflict".

&#x20; 3. RETRIEVED EXAMPLES — top-k similar TRAINING sentences with human-verified

&#x20;    GOLD entities.



\### ENTITY TYPES (exactly four)

\- "PER" (Person): A named real or fictional individual.

\- "ORG" (Organization): A named organization or organized group, such as a company, institution, agency, university, or team.

\- "LOC" (Location): A named geographical or geopolitical location, such as a country, city, region, or landmark.

\- "MISC" (Miscellaneous): A named entity that is not PER, ORG, or LOC, such as a nationality, language, event, work, award, or product.



\### DECISION ORDER (top-down; first rule that fires wins)

&#x20; STEP 1 — VERIFY presence (hard filter): keep only if the surface form occurs

&#x20;   verbatim in the SOURCE TEXT. Drop anything not in the text — a hallucination,

&#x20;   regardless of how many proposers voted. This is the only rule that overrides

&#x20;   agreement.

&#x20; STEP 2 — KEEP unless a drop rule applies. A candidate that passed Step 1 should

&#x20;   be retained unless Step 3 positively identifies it as invalid. Treat stronger

&#x20;   proposer agreement and a match to a GOLD span in the retrieved examples as

&#x20;   increasing confidence to keep; never as a threshold below which you drop.

&#x20; STEP 3 — DROP only on positive evidence, namely a candidate that is either

&#x20;   (a) absent from the source text (already handled in Step 1), or

&#x20;   (b) a generic common-noun / determiner-headed reference that names no specific

&#x20;       PER, ORG, LOC, or MISC entity, per the conventions above.

&#x20;   Do not drop a specific named entity merely because few proposers found it or it

&#x20;   is absent from the small retrieved set. When unsure whether a span is a generic

&#x20;   reference or a specific named entity, keep it.

&#x20; STEP 4 — RESOLVE type conflicts (type\_conflict = true): choose one type using,

&#x20;   in order, (a) the convention shown for a near-identical span in high-similarity

&#x20;   retrieved gold, (b) the definitions applied to this sentence, (c) the proposer

&#x20;   majority in candidate\_types.

&#x20; STEP 5 — FIX boundaries to match the gold convention where it differs, but only

&#x20;   to a form still present verbatim in the source. Never drop at this step; only

&#x20;   adjust.





\### OUTPUT FORMAT (JSON only — no markdown, no extra text)

{

&#x20; "Refiner-entities": \[

&#x20;   {

&#x20;     "surface\_form": "<exact text from source, preserving original spacing>",

&#x20;     "entity\_type": "<PER | ORG | LOC | MISC>",

&#x20;     "justification": "<1-2 sentences citing the rule, gold convention, agreement, or definition that decided it>"

&#x20;   }

&#x20; ]

}

If no valid entities exist, return {"Refiner-entities": \[]}.

"""

