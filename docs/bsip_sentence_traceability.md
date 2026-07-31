# BSIP Sentence Traceability

The Manuscript Engine assigns each non-placeholder sentence deterministic source links.

## Sentence ID Format

- Results: `SENT-RESULTS-0001`
- Discussion: `SENT-DISCUSSION-0001`
- Limitations: `SENT-LIMITATIONS-0001`
- Conclusion: `SENT-CONCLUSION-0001`
- Figure captions: `SENT-FIGURE-0001`
- Table captions: `SENT-TABLE-0001`

## Traceability Columns

`manuscript_sentence_traceability.csv` includes:

- sentence ID,
- section ID,
- sentence type,
- sentence text,
- traceability status,
- language policy status,
- source IDs,
- claim IDs,
- observation IDs,
- interpretation IDs,
- hypothesis IDs,
- evidence score IDs,
- reviewer finding IDs,
- figure IDs,
- table IDs,
- reasoning graph node IDs,
- limitations,
- metadata.

## Policy

Quantitative Results sentences must link to observation records. Interpretive Discussion sentences must link to claims, interpretations, or hypotheses. Limitation sentences must link to reviewer findings or evidence-gap nodes.

Placeholder Abstract, Introduction, and Methods sentences use `TraceabilityStatus.NOT_APPLICABLE`.
