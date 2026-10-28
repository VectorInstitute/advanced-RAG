"""Measure two claims the topic carried over from earlier, unmeasured experiments.

Claim 1: on born-digital PDFs the text layer beats OCR, and forcing OCR only
lowers quality. The universe's pages are parsed three more ways and compared
with the reference parse (Docling with OCR off, as RI #3 ingested it):

- ``ocr-auto``: Docling's own default, OCR on bitmap regions only, the text
  layer everywhere else.
- ``ocr-forced``: every page rendered to an image at 216 dpi and parsed with
  full-page RapidOCR, so no character comes from the text layer. This is what
  a scanned copy of a born-digital filing looks like at a good scan resolution.
- ``flag-forced``: Docling's ``OcrMode.FULL_PAGE`` on the original PDF, on a
  sample of pages, to check what the flag does on a page with a text layer.

Claim 2: a generic picture prompt on a small crop describes things the picture
does not contain. Pictures are binned by their share of the page, described
with three prompts, and a judge model lists each description's claims and
marks the ones the crop does not support.

Nothing here writes corpus text to the repository. Parsed folders and
descriptions stay in the local cache; the committed results hold counts and
rates.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache.paths import root as cache_root

from .pages import Corpus, page_id


#: Parse conditions compared with the reference.
CONDITIONS: tuple[str, ...] = ("ocr-auto", "ocr-forced", "flag-forced")
#: The reference: OCR off, as RI #3 ingested the filings, re-parsed from the same
#: page subsets so that every condition shares one export path and page layout.
REFERENCE = "text-layer"
#: Resolution of the rendered pages in ``ocr-forced``. Docling's OCR renders at
#: scale 3.0 (216 dpi), so the image carries what the OCR stage would see.
RASTER_DPI = 216
#: Crop-size bins, as a share of the page area: lower bounds, ascending.
AREA_BINS: tuple[tuple[str, float, float], ...] = (
    ("<0.5%", 0.0, 0.005),
    ("0.5-1%", 0.005, 0.01),
    ("1-5%", 0.01, 0.05),
    ("5-10%", 0.05, 0.10),
    (">=10%", 0.10, 1.01),
)

#: A third prompt: the generic request plus the domain prompt's grounding lines,
#: with no domain content, to separate grounding from domain knowledge.
GROUNDED_GENERIC_PROMPT = (
    "Describe this image in a few sentences. Use only what the image shows. Do not add "
    "context from outside it. If it is a logo, icon, signature, or decoration, say so in "
    "one sentence."
)

UNSUPPORTED_PROMPT = """You are checking a description of the image above against the image.

Description:
<<<
{description}
>>>

Split the description into its factual claims: each stated object, label, number,
trend, relation, or identification (a company, product, or place) is one claim.
For each claim decide:
- "supported": the image shows it;
- "unsupported": the image does not show it, or shows something different;
- "generic": a hedge or framing with no checkable content ("the image appears to be").

Return one JSON object and nothing else:
{{"claims": [{{"claim": "<short paraphrase>", "verdict": "supported|unsupported|generic"}}]}}"""


def replication_root() -> Path:
    """Return where the extra parses are written, one folder per condition."""
    return Path(cache_root()) / "ingestion" / "financebench-replication"


def condition_corpus(base: Corpus, condition: str) -> Corpus:
    """Return a corpus that reads one condition's Docling folders."""
    return Corpus(base.pdf_dir, replication_root() / condition)


def area_bin(area: float) -> str:
    """Return the name of the crop-size bin an area share falls in."""
    for name, low, high in AREA_BINS:
        if low <= area < high:
            return name
    raise ValueError(f"area share out of range: {area}")


# --------------------------------------------------------------------------- parsing


def remap_pages(exported: dict[str, Any], pages: Sequence[int]) -> dict[str, Any]:
    """Return a Docling export of a page subset renumbered to the source's pages.

    Parameters
    ----------
    exported : dict
        ``DoclingDocument.export_to_dict()`` of a PDF built from ``pages``.
    pages : sequence of int
        The zero-based source pages, in the order they were copied.

    Returns
    -------
    dict
        The same export with every ``page_no`` (one-based) pointing at the
        source page, so the topic's page identifiers stay valid.
    """
    mapping = {index + 1: page + 1 for index, page in enumerate(pages)}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "page_no" in node and isinstance(node["page_no"], int):
                node["page_no"] = mapping.get(node["page_no"], node["page_no"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    out: dict[str, Any] = json.loads(json.dumps(exported))
    walk(out)
    out["pages"] = {str(mapping[int(k)]): v for k, v in exported.get("pages", {}).items()}
    for key, value in out["pages"].items():
        value["page_no"] = int(key)
    return out


def raster_pdf(source: Path, pages: Sequence[int], dest: Path, dpi: int = RASTER_DPI) -> None:
    """Write an image-only PDF of the given zero-based pages at ``dpi``."""
    import pypdfium2 as pdfium  # noqa: PLC0415

    document = pdfium.PdfDocument(str(source))
    images = [document[p].render(scale=dpi / 72).to_pil().convert("RGB") for p in pages]
    images[0].save(dest, save_all=True, append_images=images[1:], resolution=float(dpi))


def subset_pdf(source: Path, pages: Sequence[int], dest: Path) -> None:
    """Write a PDF holding only the given zero-based pages, text layer kept."""
    import pypdfium2 as pdfium  # noqa: PLC0415

    document = pdfium.PdfDocument(str(source))
    out = pdfium.PdfDocument.new()
    out.import_pages(document, list(pages))
    out.save(str(dest))


@lru_cache(maxsize=4)
def converter(condition: str) -> Any:
    """Return the Docling converter for one condition, table structure on as in RI #3.

    Cached, because building one loads the layout, table, and OCR models, which
    costs more than parsing a document's few universe pages.
    """
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions  # noqa: PLC0415
    from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415

    if condition == "ocr-auto":
        options = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
    elif condition in ("ocr-forced", "flag-forced"):
        options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            ocr_options=RapidOcrOptions(mode=OcrMode.FULL_PAGE, lang=["en"]),
        )
    elif condition == "text-layer":
        options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def parse_condition(corpus: Corpus, doc_id: str, pages: Sequence[int], condition: str, workdir: Path) -> Path:
    """Parse one document's pages under one condition and write its folder.

    The folder holds ``document.json`` (renumbered to the source pages) and
    ``pages/pNNN.md``, the two files the topic reads.
    """
    folder = replication_root() / condition / doc_id
    if (folder / "document.json").is_file():
        return folder
    workdir.mkdir(parents=True, exist_ok=True)
    staged = workdir / f"{doc_id}-{condition}.pdf"
    if condition == "ocr-forced":
        raster_pdf(corpus.pdf(doc_id), pages, staged)
    else:
        subset_pdf(corpus.pdf(doc_id), pages, staged)
    document = converter(condition).convert(staged).document
    exported = remap_pages(document.export_to_dict(), pages)
    (folder / "pages").mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(pages):
        text = document.export_to_markdown(page_no=index + 1)
        (folder / "pages" / f"p{page:03d}.md").write_text(text, encoding="utf-8")
    (folder / "document.json").write_text(json.dumps(exported, ensure_ascii=False) + "\n", encoding="utf-8")
    staged.unlink(missing_ok=True)
    return folder


def pages_by_doc(page_ids: Iterable[str]) -> dict[str, list[int]]:
    """Group page identifiers by document, pages ascending."""
    from .pages import split_page_id  # noqa: PLC0415

    out: dict[str, list[int]] = {}
    for identifier in page_ids:
        doc_id, page = split_page_id(identifier)
        out.setdefault(doc_id, []).append(page)
    return {d: sorted(p) for d, p in sorted(out.items())}


# ------------------------------------------------------------------- text fidelity

#: A number standing alone: a digit run glued to letters (an OCR "2O21") is not one.
_NUMBER = re.compile(r"(?<![A-Za-z0-9])\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?(?![A-Za-z0-9])")
_WORD = re.compile(r"[A-Za-z0-9$%().,-]+")


def normalise(text: str) -> str:
    """Return text with markdown table rules, emphasis, and whitespace runs removed."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\|?\s*-{3,}\s*\|", " ", text)
    text = text.replace("|", " ").replace("**", "").replace("#", " ")
    return re.sub(r"\s+", " ", text).strip()


def numbers(text: str) -> Counter[str]:
    """Return the multiset of numeric tokens, parentheses, signs, and commas dropped."""
    found: Counter[str] = Counter()
    for match in _NUMBER.findall(text):
        token = match.strip("()$").replace(",", "").rstrip("%")
        if any(ch.isdigit() for ch in token):
            found[token] += 1
    return found


def multiset_recall(reference: Counter[str], candidate: Counter[str]) -> tuple[int, int]:
    """Return how many reference tokens the candidate holds, and the reference size."""
    return sum((reference & candidate).values()), sum(reference.values())


def word_accuracy(reference: str, candidate: str) -> float:
    """Return the share of reference words aligned to an identical candidate word."""
    ref = _WORD.findall(reference.lower())
    cand = _WORD.findall(candidate.lower())
    if not ref:
        return 1.0 if not cand else 0.0
    matcher = SequenceMatcher(None, ref, cand, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(ref)


def page_fidelity(reference: str, candidate: str) -> dict[str, Any]:
    """Compare one page's text under a condition with the text-layer parse."""
    ref, cand = normalise(reference), normalise(candidate)
    hit, total = multiset_recall(numbers(ref), numbers(cand))
    extra = sum((numbers(cand) - numbers(ref)).values())
    return {
        "ref_chars": len(ref),
        "chars": len(cand),
        "identical": ref == cand,
        "word_accuracy": round(word_accuracy(ref, cand), 4),
        "numbers_ref": total,
        "numbers_kept": hit,
        "numbers_extra": extra,
    }


def table_cells(exported: Mapping[str, Any], page: int) -> Counter[str]:
    """Return the multiset of numeric table cells Docling found on a one-based page."""
    found: Counter[str] = Counter()
    for table in exported.get("tables", []):
        prov = table.get("prov") or []
        if not prov or prov[0].get("page_no") != page:
            continue
        for cell in table.get("data", {}).get("table_cells", []):
            text = str(cell.get("text", "")).strip()
            tokens = numbers(text)
            if len(tokens) == 1 and sum(tokens.values()) == 1:
                found.update(tokens)
    return found


def table_count(exported: Mapping[str, Any], page: int) -> int:
    """Return how many tables Docling found on a one-based page."""
    return sum(1 for t in exported.get("tables", []) if (t.get("prov") or [{}])[0].get("page_no") == page)


def load_export(folder: Path) -> dict[str, Any]:
    """Return a Docling folder's ``document.json`` as a dict."""
    loaded: dict[str, Any] = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    return loaded


def summarise_unsupported(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return claim counts and unsupported shares for a group of judged descriptions.

    ``any_unsupported`` is the share of descriptions with at least one
    unsupported claim; ``claim_share`` is unsupported claims over checkable
    (supported plus unsupported) claims.
    """
    rows = [r for r in records if r.get("claims") is not None]
    supported = sum(r["supported"] for r in rows)
    unsupported = sum(r["unsupported"] for r in rows)
    return {
        "n": len(rows),
        "claims_per_description": round((supported + unsupported) / len(rows), 2) if rows else None,
        "any_unsupported": round(sum(r["unsupported"] > 0 for r in rows) / len(rows), 4) if rows else None,
        "claim_share": round(unsupported / (supported + unsupported), 4) if supported + unsupported else None,
    }


def count_verdicts(found: Mapping[str, Any]) -> dict[str, Any]:
    """Return supported, unsupported, and generic counts from a judge's parsed reply."""
    claims = found.get("claims")
    if not isinstance(claims, list):
        return {"claims": None, "supported": 0, "unsupported": 0, "generic": 0}
    verdicts = Counter(str(c.get("verdict", "")).strip().lower() for c in claims if isinstance(c, Mapping))
    return {
        "claims": len(claims),
        "supported": verdicts.get("supported", 0),
        "unsupported": verdicts.get("unsupported", 0),
        "generic": verdicts.get("generic", 0),
    }


__all__ = [
    "AREA_BINS",
    "CONDITIONS",
    "GROUNDED_GENERIC_PROMPT",
    "RASTER_DPI",
    "UNSUPPORTED_PROMPT",
    "area_bin",
    "condition_corpus",
    "count_verdicts",
    "load_export",
    "multiset_recall",
    "normalise",
    "numbers",
    "page_fidelity",
    "page_id",
    "pages_by_doc",
    "parse_condition",
    "remap_pages",
    "replication_root",
    "summarise_unsupported",
    "table_cells",
    "table_count",
    "word_accuracy",
]
