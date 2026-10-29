# Evaluation: where retrieval fails on long documents

## Method

`eval_rag.py` builds a synthetic 60-page service manual for a fictional
aircraft ("Kestrel-9"): 12 sections x 5 pages. Six component types (pressure
relief valve, coolant pump, check valve, filter assembly, actuator, sensor
module) are shared across sections, and every (section, component) pair gets
different spec values. Spec sentences never name their section:

> "The pressure relief valve is rated at 3825 psi."

24 questions ask for a spec in a named section ("According to the manual,
what is the pressure relief valve rated at in the Hydraulic System?").
Ground truth = the page containing that section's spec sentence.

Both arms use the exact app configuration: `all-MiniLM-L6-v2` embeddings,
`RecursiveCharacterTextSplitter` (1200 chars / 300 overlap), cosine
similarity, k=5. The only difference is whether each chunk is prepended
with `[Kestrel-9 Service Manual | Section: X | Page N of 60]` before
embedding.

Seed is fixed (42), so `python eval_rag.py make-data && python eval_rag.py run`
reproduces these numbers exactly. Measured with the quantized ONNX export of
`all-MiniLM-L6-v2` (same weights as the sentence-transformers default).

## Results

| Retriever                                    | Hit@1 | Hit@5 | MRR   |
|----------------------------------------------|-------|-------|-------|
| Baseline (bare chunks)                       | 2/24 (8.3%)  | 15/24 (62.5%) | 0.278 |
| Context-enriched chunks                      | 8/24 (33.3%) | 24/24 (100%)  | 0.581 |

Baseline questions where the correct page never made the top 5 (9 of 24):

| Gold page | Question (abbreviated)                                    | Enriched rank |
|-----------|-----------------------------------------------------------|---------------|
| 2         | pressure relief valve rating, Hydraulic System            | 3             |
| 7         | coolant pump rating, Cooling System                       | 4             |
| 22        | actuator rating, Landing Gear                             | 4             |
| 32        | pressure relief valve rating, Navigation Module           | 1             |
| 33        | coolant pump inspection interval, Navigation Module       | 3             |
| 42        | check valve rating, Thermal Control                       | 5             |
| 43        | filter assembly inspection interval, Thermal Control      | 2             |
| 48        | actuator inspection interval, Propulsion Core             | 3             |
| 58        | pressure relief valve inspection interval, Structural Frame | 3           |

7 of the 9 failures are past page 20: the deeper into the document, the
more confounder chunks exist, and the worse bare retrieval gets.

## Interpretation

- The failure is *disambiguation*, not *similarity*: the retriever finds
  near-identical spec sentences easily, it just can't tell which section
  they belong to. A one-line context header gives the embedding that
  signal, and every top-5 miss disappears.
- Hit@1 only reaches 33% because confounders from the *correct* section
  still crowd the top ranks. A cross-encoder rerank over the top 5-10 is
  the obvious next step.
- These numbers measure retrieval, not answer correctness. A generation
  eval through Ollama is scaffolded in `eval_rag.py --gen`.

## Limitations

- Synthetic dataset isolating one failure mode; not a general benchmark.
- Section titles in real PDFs are noisier than `detect_section()` assumes.
- The running-summary ingest path is implemented but not measured here.
- Single seed, single embedding model, single chunking config.
