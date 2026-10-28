"""Parse, enrich, or embed the page: the visual-documents special topic.

The modules here hold the code the two notebooks in this folder run, and the
build script that produced the cached reference run:

- :mod:`.pages` chooses the page universe from the FinanceBench filings RI #3
  ingests, renders pages and picture crops, and reads Docling's page text.
- :mod:`.vlm` sends text and image requests through the Vector proxy, throttled,
  cached, and metered.
- :mod:`.enrich` writes picture descriptions and table summaries, the harness
  approach, and folds them into the chunk text.
- :mod:`.questions` builds the figure-only and table-only question set.
- :mod:`.visual` embeds page images with ColModernVBERT and searches them by
  late interaction.
- :mod:`.arms` turns every arm into a page ranking and a reader input.
- :mod:`.scoring` scores pages and answers and prices each arm.
"""
