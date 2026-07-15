# PReFER — Multi-Proposer + Refiner Scientific NER

```
sentence ──► \[ P1  P2  P3  P4 ]  ──►  union + position-aware dedup  ──►  Refiner  ──►  final entities
             (parallel proposers)      (merged candidate set)          (1 call)
```


## Zero-Shot (ZS) vs. Few-Shot (FS)

Controlled by `RETRIEVAL\_MODE`. This decides whether the Refiner sees in-context (ICL) gold examples.

|`RETRIEVAL\_MODE`|Type|Behaviour|
|-|-|-|
|`"no"`|**Zero-Shot (ZS)**|Bare Refiner prompt: no retrieval, no training examples. **(default)**|
|`"hybrid"`|**Few-Shot (FS)**|Retrieve from both dense and BM25, fuse with Reciprocal Rank Fusion (RRF).|


\---

## Key parameters

|Parameter|Default|Applies to|
|-|-|-|
|`TEMPERATURE\_PROPOSER`|`0.3`|Proposer|
|`MAX\_TOKENS\_PROPOSER`|`2000`|Proposer (room for reasoning + JSON)|
|`TEMPERATURE\_REFINER`|`0.3`|Refiner|
|`MAX\_TOKENS\_REFINER`|`900`|Refiner|
|`TOP\_P\_REFINER`|`0.9`|Refiner|

### Retrieval (FS modes only)

|Parameter|Default|Meaning|
|-|-|-|
|`RETRIEVAL\_MODE`|`"hybrid"`|ZS/FS selector (see table above).|
|`TOP\_K\_RETRIEVAL`|`5`|Number of ICL examples per sentence.|
|`EMBEDDING\_BACKEND`|`"sentence\_transformers"`|`"sentence\_transformers"` (local) or `"ionos"` (remote).|
|`EMBED\_BATCH\_SIZE`|`64`|Embedding batch size.|
|`RRF\_K`|`60`|RRF damping constant (hybrid).|
|`HYBRID\_CANDIDATE\_POOL`|`50`|Candidates pulled from each retriever before fusion.|
|`CLEAN\_TEXT\_FOR\_RETRIEVAL`|`True`|Strip citations/URLs for BM25 matching only.|


## Data layout

Expects SciER-style JSONL with one sentence per line and columns:

* `doc\_id` — document identifier
* `sentence` — the input text
* `ner` — gold labels as `\[\[surface\_form, entity\_type], ...]`

```
data/
└── SciER/
    ├── test.jsonl
    └── train.jsonl   # needed only for few-shot (FS) retrieval modes
```

\---

## Install

```bash
pip install openai pandas numpy
# few-shot retrieval only:
pip install sentence-transformers rank\_bm25
```

## Run

```bash
export OPENAI\_API\_KEY="your-key"
# optional: export OPENAI\_BASE\_URL="https://your-endpoint/v1"
python prefer\_sciner.py
```

`main()` loads the test split, runs the full pipeline, saves results, and prints the evaluation.

Evaluation

Precision / Recall / F1 (per class + micro / macro / weighted) are reported for every 
proposer, the merged Union set, and the final Refiner output, under both exact and partial
matching. 

