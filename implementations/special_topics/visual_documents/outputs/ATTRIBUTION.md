# Attribution for the results in this directory

The four JSON files hold identifiers, counts, and metrics from the reference run of
`01_parse_enrich_or_embed.ipynb`:

- `question_set.json`: the question set's counts, its generation funnel, and each
  question's identifier, group, gold page identifiers, and picture or table identifiers.
- `results.json`: page retrieval and answer accuracy per arm and question group, the
  retrieval loss against each arm's oracle, and the paired tests.
- `costs.json`: enrichment calls and tokens, index sizes and embedding times, and reader
  tokens per question, with list-price estimates for the paid default models.
- `description_audit.json`: the share of sampled picture descriptions a second model
  flagged against the picture.

No file carries question text, answer text, description text, or filing text. A page
appears only as `<FinanceBench doc_name>#p<zero-based page>`, and a picture or table as
`<doc_name>|<Docling reference>`. The question text, the descriptions, the page index,
and every model response stay in the local artifact and response caches.

The figures derive from FinanceBench, which the Hugging Face dataset card licenses
CC BY-NC 4.0, so they carry its attribution:

```text
FinanceBench: A New Benchmark for Financial Question Answering, Islam,
Kannappan, Kiela, Qian, Scherrer, and Vidgen, 2023, Patronus AI.
https://arxiv.org/abs/2311.11944
```

Changes made: 1,000 pages of 49 of the 84 filings the 150 questions cite were rendered,
parsed with Docling, described and summarised by a vision model, embedded, and used to
generate and check new questions. FinanceBench's own questions on those filings appear
as controls, by identifier. See `../../../table_structured_reasoning/CORPUS.md`.
