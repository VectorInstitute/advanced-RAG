"""FinanceBench: 150 analyst questions over SEC 10-K filings, pinned and checked.

The loader reads the 150-question open sample and the 84 filings those questions
cite. Every file is addressed by the commit of ``patronus-ai/financebench`` it
was read from and by a SHA-256 digest, so a changed upstream file fails here
instead of moving a figure. The loader never reads a moving reference such as
``main``.

Gold evidence is page-level. Each question lists one or more evidence entries
with a page number, and those numbers are zero-based: the smallest in the sample
is 0, which is the PDF's first page. ``PAGE_BASE`` states that so a consumer
does not have to rediscover it.

This module is not registered in ``corpora.SOURCES``. The registry's test
requires every entry to permit commercial use, and this corpus is licensed
non-commercial. Its files are fetched into the local corpus cache and never
committed.

The licence record states what the sources say. The terms come from the Hugging
Face dataset card, because the GitHub repository that hosts the annotation files
and the PDFs states none.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aieng.advanced_rag.retrieval.corpus import Query

from .sources import Licence, PinnedSource, cache_dir, fetch


#: The commit of ``patronus-ai/financebench`` every file below was read at.
FINANCEBENCH_COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"

#: Evidence page numbers in the annotations start at 0.
PAGE_BASE = 0

_RAW = (
    f"https://raw.githubusercontent.com/patronus-ai/financebench/{FINANCEBENCH_COMMIT}"
)

FINANCEBENCH_LICENCE = Licence(
    name="CC BY-NC 4.0",
    url="https://creativecommons.org/licenses/by-nc/4.0/",
    permits_commercial_use=False,
    attribution=(
        "FinanceBench: A New Benchmark for Financial Question Answering, Islam, "
        "Kannappan, Kiela, Qian, Scherrer, and Vidgen, 2023, Patronus AI. "
        "https://arxiv.org/abs/2311.11944"
    ),
    notes=(
        "The Hugging Face dataset card for PatronusAI/financebench (revision "
        "e04404e3, modified 2024-11-17) states license: cc-by-nc-4.0. The GitHub "
        "repository that hosts the annotation files and the PDFs carries no "
        "licence file and no licence statement. The filings are SEC public "
        "filings by public companies. TAT-QA, licensed CC BY 4.0 per its README, is an "
        "alternative that permits commercial use."
    ),
)

#: The 150-question open sample, one JSON object per line.
FINANCEBENCH_OPEN_SOURCE = PinnedSource(
    name="financebench-open-source",
    url=f"{_RAW}/data/financebench_open_source.jsonl",
    sha256="a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877",
    licence=FINANCEBENCH_LICENCE,
)

#: Company, sector, filing type, period, and source link for 361 filings.
FINANCEBENCH_DOCUMENT_INFORMATION = PinnedSource(
    name="financebench-document-information",
    url=f"{_RAW}/data/financebench_document_information.jsonl",
    sha256="1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575",
    licence=FINANCEBENCH_LICENCE,
)

#: SHA-256 of each of the 84 filings the 150 questions cite, by document name.
PDF_SHA256: dict[str, str] = {
    "3M_2018_10K": "faf10c53fc6f1a8aa346ec3db79145e3ae6fbb3753c8064924e174d0a973934d",
    "3M_2022_10K": "d5cf549543a24b04228fd2af979ff2ca94cf64fb008a789340cb9117fbcfde5d",
    "3M_2023Q2_10Q": "3e66115793459f42185feeeab4e7f07dd27ab6dd465d0694ce415e861029cf93",
    "ACTIVISIONBLIZZARD_2019_10K": "017f1bbc1e37f2945a41fa2b5cf39d822e1e9d435a2c2b9c6b1cc76cd114a2df",
    "ADOBE_2015_10K": "f9ddba652f19e20752d98b1b4cedcebcb1a9a6d11dd23ab2e2f00f009a164c24",
    "ADOBE_2016_10K": "655153cdaed14d7093e98ff75595bf850a0d2299fa851ccf5b344ebc2d5038ae",
    "ADOBE_2017_10K": "ace32580369138998de84a0d0e24943766aa8374a17b823fdfae980b9c1458bd",
    "ADOBE_2022_10K": "ce8c716abac26bfa38ae6b71d703112852c8591a4bb19f7c1177beb1dfc87ad2",
    "AES_2022_10K": "9c5de2227534c4d6bf3d805f635bad5d82eddf8e4183a85a3d1311b48b50291d",
    "AMAZON_2017_10K": "f8af08ff6c4147351ec3a6e5e2c74e104678ce68fb0d6d476b1e43f685c09a3f",
    "AMAZON_2019_10K": "d7df8c6cf3e351a8d68c7473be4159723d545260a32dbc82a8dd11c254c5584a",
    "AMCOR_2020_10K": "da158d5c68835374bcc52205663aa7653af1c2a6b31b09ab775c6e02a40da026",
    "AMCOR_2022_8K_dated-2022-07-01": "1de32e72c265486469f3b1c7cbfbe064a27055e60f7b43a81439986dda9025ad",
    "AMCOR_2023_10K": "84d7a0291a12f4e68352750dccda3754c78eefdf06e0b5462046357a1ed53332",
    "AMCOR_2023Q2_10Q": "71c39c3f338fc81a99e128e8e871e265818657670a7d29a62c454c905f94524c",
    "AMCOR_2023Q4_EARNINGS": "d155b7808e78167831648f476b7e3578e2d98645dee30862bbef4ff50fa24bf4",
    "AMD_2015_10K": "105fff1c4af84a3d974b0f3fa224122e782aedca60fa3609f2789c80549dd8f6",
    "AMD_2022_10K": "a3bd74088fae0ad4aa03d04998a1f4c64fd0b3c693841601b0ac49b46cf5c1f4",
    "AMERICANEXPRESS_2022_10K": "d3bbc7ab23d6160e07eab50a359e4d68c40efff9fde3baa168e7c40e06f568e6",
    "AMERICANWATERWORKS_2020_10K": "6b43f1ed70b67c06483d28f8054270d597bce8507f298af2b1b6d9356d728602",
    "AMERICANWATERWORKS_2021_10K": "701528ae6d14a77229468a829ee18f01ed439d5be012d0a915d2683335051029",
    "AMERICANWATERWORKS_2022_10K": "aea336531f1f8b10ecee1dd61066aab29794504f1f4a1407c857bd1e20d7da40",
    "BESTBUY_2017_10K": "4570187fa697f252e08c8f99c409446fbcde05b08358851d34d4cb1c8bd1cec8",
    "BESTBUY_2019_10K": "d3228e9acd2ce498b97a9d2cb99c2330b9a04687b7947908646d205b3eadf30d",
    "BESTBUY_2023_10K": "7ca465d31a08be29e8c08d756644d33d818c7254a1215720f265227005cda696",
    "BESTBUY_2024Q2_10Q": "cf9efb922bdd28d726d7e1605c5fa1f263dda72e033989e3bc9ee3a3bb356c03",
    "BLOCK_2016_10K": "5ef866a19b1110dcdbd1b0bc3c513a04e3d05322384c754cf394ad63f9af8191",
    "BLOCK_2020_10K": "c19a9fb25db3270e3a908c202855f4a54f000a06bf19a8b284331810228646fe",
    "BOEING_2018_10K": "6f1ace34b3b01778ad7d92d43ccd2f76513d5159ef48fbc07b510296cf2d07dc",
    "BOEING_2022_10K": "09285ff7ee737d3302977104aa05cc53c9dfb0161f3461a89b583dc38b9f2ab6",
    "COCACOLA_2017_10K": "b57c75cf20916904f31c6aa7644d6c110f019adf7c76782c7d0ede7b95c65ac1",
    "COCACOLA_2021_10K": "46a7059b9da795cd1ea9862d5b4f82a21e9b72e799351927a11f5cb4c528b592",
    "COCACOLA_2022_10K": "9f91bb914d94d014cfca96e1021c932d5476d8b8ac2a933767bac4c438fd7a71",
    "CORNING_2020_10K": "ef471ab9ff6067f117977df6df8b4b5007066286b733981475ac2ccf48d91e59",
    "CORNING_2021_10K": "8023fac91ad417f996b692c506ee0e097effd091e698275288e4126cb8a2020e",
    "CORNING_2022_10K": "a63f102f045826b38ad21a606c33710f62932646fd7c0f1852ce411a4397a3ee",
    "COSTCO_2021_10K": "29e05b3862cf6ffa6cfda34ac82a9fa0fe76b67338c5924da3325c2ee3b08748",
    "CVSHEALTH_2018_10K": "bdc800e3f52a3e1d7a79c0973af3e94122fa08a76e335ea8305df60a09e1a0dc",
    "CVSHEALTH_2022_10K": "183c196a41793cd50002ff6b88f8c6d32e4fcb1117ab490a732414e1208cda8f",
    "FOOTLOCKER_2022_8K_dated-2022-05-20": "865f22331a5943a85e53e38d1753addbeb6c42270853cddd0abc73ab23d150e9",
    "FOOTLOCKER_2022_8K_dated_2022-08-19": "70fd2d5bfc6955424196b372a5273ffbc74d2f0153f291e10e73edc37962a14e",
    "GENERALMILLS_2019_10K": "e8120ce31b22136e15c14115cdfbf3ddc548e58f15b6febc350c422a4c12d9db",
    "GENERALMILLS_2020_10K": "626dd53f9b52e5895b7a7073e1c491008769970609c10e2f6cf8247ed85d6fae",
    "GENERALMILLS_2022_10K": "27f2cc507ffc056de3b679d124d51b80bd043ce7b8e867c59e2cdfe1d2ea5de3",
    "JOHNSON_JOHNSON_2022_10K": "dba7d6cc97734abd6dcbc1a6a34f82b5d228e60d0a1a32425a90404fd4d3e290",
    "JOHNSON_JOHNSON_2022Q4_EARNINGS": "7ae6bf6f9a998d19e23fd796b58b0f2015085c74712253b78fb90174c12436a7",
    "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30": "0601a4b9f15e4400ced1b2ccb81c309cb66844fbac6b975a76dc490501bea95a",
    "JOHNSON_JOHNSON_2023Q2_EARNINGS": "db1f3419d309440e644dc3cdfbb74758e67734dd181e5dbabc7479d0ab92d026",
    "JPMORGAN_2021Q1_10Q": "62518c05b02265dc13ca257d9ec0be0aec4449dad53db0c77d93444e7e2cfe97",
    "JPMORGAN_2022_10K": "c269899950f36edfa496e3c8447a51d86d0a206442a96dc24dcd9317d55b957c",
    "JPMORGAN_2022Q2_10Q": "7d715f552d4eeb6b2a1ea8ae02237b5767306907e761270ac23d74ac5b7e44b5",
    "JPMORGAN_2023Q2_10Q": "eef2d9c1fdb5dc82be48166487a9e451c8544aec932a01fb06309ea7cd7122f5",
    "KRAFTHEINZ_2019_10K": "df1340e7b002e5f412d4c9073292d0f48a2988fd347d5a4f44d017b9680fbae2",
    "LOCKHEEDMARTIN_2020_10K": "1e68051a649270779fc5cb68806612a28d7d81288ed75dd37aa1858717bf55a0",
    "LOCKHEEDMARTIN_2021_10K": "d2486265fedea887bcc621f0b2dbb9670e66fe05203e1378b8e650c9564732fb",
    "LOCKHEEDMARTIN_2022_10K": "9a2317300d0f1e2623ba291fe9ec38fc779c1d1ad989dcfe6f862b77435c6923",
    "MGMRESORTS_2018_10K": "1b540737f91716aaa23b062499e6cab9c51d258e151f37eb8c65646bb588a0da",
    "MGMRESORTS_2020_10K": "98a9a55baf838309d16a730fd63efbcd146be1911b61abb3658b1774f360b099",
    "MGMRESORTS_2022_10K": "130b96c1309c97aaedcabee2fa5d5421e54491e618273bd33c3b1aeddad323ec",
    "MGMRESORTS_2022Q4_EARNINGS": "fe9126b17c3656e50d53741b13cb5fb233d4adaf922e4d84072b1d5e564fbfbd",
    "MGMRESORTS_2023Q2_10Q": "09c02fd40073570ae0826d707674154a9cd55a895e94096a82fcf0a4447ea64d",
    "MICROSOFT_2016_10K": "f3555bd5ea1802770cc8f2a7ca766392e16fe6504f18688db681d852029404dc",
    "MICROSOFT_2023_10K": "e534af371c824aab19b8d01f8bc0e9d1e0a3fb418a33119c5224a53d20a0d37b",
    "NETFLIX_2015_10K": "b22ca874ef08f383bd0b1071677d44f4a242af668f855f00dc9c6d4ac94148e2",
    "NETFLIX_2017_10K": "981a69bba010dd1709c74ccec9ca26c61635c02a4f22b76345d26021e88077d6",
    "NIKE_2018_10K": "31de66f1e37cabf1785e0ffa66955bf003bf2a9dcf3f4ebe93732d063b7b6ebe",
    "NIKE_2019_10K": "7dbdf1bcbf37dd6d6e694ce95bcb23cd6943d08cc9cd1e5ee9832ed8f7eebdc6",
    "NIKE_2021_10K": "73b54e5e0b1b0b33e51ac314f6d9a9db0bcc286269feb24a3445a5f1ff0f7789",
    "NIKE_2023_10K": "51f5a0fc6ede20de4f5cfdeae8791a57bdd65c95db2ceb263eb120696ff90172",
    "PAYPAL_2022_10K": "0c1dd6cd170261c6bd778de382a211a56a3bb1e009a429246370c97fa0defab3",
    "PEPSICO_2021_10K": "f2c7cf1969c7f6dfc8c0596824e1d6b3cc70f662ee2dd06be84bb27af5635364",
    "PEPSICO_2022_10K": "7d8a9f3374e429c58f101319e2da8a0e601a89859e1e4f6854163dd5f8ce6b47",
    "PEPSICO_2023_8K_dated-2023-05-05": "e8591d6f83d810a0f9edfc540a1a21aa6269a6de4c756b5eb808d5dbbea330ed",
    "PEPSICO_2023_8K_dated-2023-05-30": "8e8f99f270b065e105636f54b2684f4c4170b3fc80845ae2f393bf66f3291d2c",
    "PEPSICO_2023Q1_EARNINGS": "a705483ea6c548a2f67c421bc24ced391a6beb8487600b44b54e9ae2aca1f31c",
    "PFIZER_2021_10K": "ff3307d5f633c5dbe59a976f5ff5e086af04ba5a5135f2b6411c89538a22a665",
    "Pfizer_2023Q2_10Q": "bf6c059a1dd71bd56581c94bce650bd85b56de2f4d0a25b45f05804a8bd6684d",
    "ULTABEAUTY_2023_10K": "cdf2ebcfbc2eb9ecf880126d38b186cac1f683dae5bcb50cad15176bfc449ff4",
    "ULTABEAUTY_2023Q4_EARNINGS": "c7df6210a627042e231c9579982648a43f3ff8b630b4020540330c2a85e0d781",
    "VERIZON_2021_10K": "f7fcb508f920e3df9a07bc858a6d9f0d3cd1e976a46efe72375c9ffb983822ac",
    "VERIZON_2022_10K": "403d7346e8a059472863d1789527f16e5127a61c2f6c4429012e6227f27576d7",
    "WALMART_2018_10K": "661512df698c895a4a5ab3d4ab879fd8ffc8f17a9038e27bcb208401982b6c15",
    "WALMART_2019_10K": "f1d1b052c2b68c5d68e3575de7069dc47914f03abb02771d2629b10bf7d9e377",
    "WALMART_2020_10K": "0f9e2ebc9432caef54fe932ec91ccc6fea3eac89f370073462ae002eabb71a34",
}


def pdf_source(doc_name: str) -> PinnedSource:
    """Return the pinned source for one cited filing.

    Parameters
    ----------
    doc_name : str
        A name from the annotations, such as ``3M_2018_10K``.

    Returns
    -------
    PinnedSource
        The filing at the pinned commit.

    Raises
    ------
    KeyError
        If no question cites ``doc_name``.
    """
    if doc_name not in PDF_SHA256:
        raise KeyError(
            f"no question cites {doc_name!r}; known: {len(PDF_SHA256)} filings"
        )
    return PinnedSource(
        name=doc_name,
        url=f"{_RAW}/pdfs/{doc_name}.pdf",
        sha256=PDF_SHA256[doc_name],
        licence=FINANCEBENCH_LICENCE,
    )


@dataclass(frozen=True)
class FinanceQuery:
    """One FinanceBench question and its page-level evidence.

    Parameters
    ----------
    query_id : str
        The dataset's own identifier, such as ``financebench_id_03029``.
    question : str
        The question text.
    doc_id : str
        The cited filing's name. Retrieval scopes to it.
    category : str
        ``question_type``: metrics-generated, domain-relevant, or novel-generated.
    reasoning : str
        ``question_reasoning``, which the dataset leaves empty for some questions.
    answer : str
        The reference answer.
    gold_pages : tuple of int
        Zero-based pages that carry evidence, sorted and without repeats.
    evidence_texts : tuple of str
        The annotated evidence passage for each evidence entry.
    """

    query_id: str
    question: str
    doc_id: str
    category: str
    reasoning: str
    answer: str
    gold_pages: tuple[int, ...]
    evidence_texts: tuple[str, ...]

    @property
    def is_answerable(self) -> bool:
        """Return True: every question in the open sample has evidence."""
        return True

    def as_query(self) -> Query:
        """Return the retrieval query the harness's retrievers take.

        Retrievers read the question and the scope only. The returned query has
        no answer spans, so it is not ``answerable`` in CUAD's sense, and
        judgments come from ``gold_pages`` instead.
        """
        return Query(
            query_id=self.query_id,
            doc_id=self.doc_id,
            category=self.category,
            question=self.question,
        )


@dataclass(frozen=True)
class FinanceBenchPaths:
    """Where the verified files are.

    Parameters
    ----------
    open_source : Path
        The annotation file.
    document_information : Path
        The filing metadata file.
    pdfs : dict
        Document name to the verified PDF.
    """

    open_source: Path
    document_information: Path
    pdfs: dict[str, Path]


def parse_queries(lines: Iterable[str]) -> list[FinanceQuery]:
    """Parse annotation lines into questions.

    Parameters
    ----------
    lines : iterable of str
        One JSON object per non-empty line.

    Returns
    -------
    list of FinanceQuery
        In file order.

    Raises
    ------
    ValueError
        If a question has no evidence entry, or its evidence names a different
        filing than the question does.
    """
    queries = []
    for line in lines:
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        evidence = record.get("evidence") or []
        if not evidence:
            raise ValueError(f"{record['financebench_id']} has no evidence entry")
        stray = {e["doc_name"] for e in evidence} - {record["doc_name"]}
        if stray:
            raise ValueError(
                f"{record['financebench_id']} cites evidence in {sorted(stray)}, "
                f"not only in {record['doc_name']}"
            )
        queries.append(
            FinanceQuery(
                query_id=str(record["financebench_id"]),
                question=str(record["question"]),
                doc_id=str(record["doc_name"]),
                category=str(record.get("question_type") or ""),
                reasoning=str(record.get("question_reasoning") or ""),
                answer=str(record.get("answer") or ""),
                gold_pages=tuple(
                    sorted({int(e["evidence_page_num"]) for e in evidence})
                ),
                evidence_texts=tuple(str(e["evidence_text"]) for e in evidence),
            )
        )
    return queries


def load_queries(path: Path | None = None) -> list[FinanceQuery]:
    """Load the 150 questions, fetching and verifying the annotation file.

    Parameters
    ----------
    path : Path or None, optional
        An annotation file already on disk. When omitted the pinned file is
        fetched into the corpus cache and verified.

    Returns
    -------
    list of FinanceQuery
        The open sample.
    """
    source = path if path is not None else fetch(FINANCEBENCH_OPEN_SOURCE)
    return parse_queries(Path(source).read_text(encoding="utf-8").splitlines())


def fetch_financebench(
    doc_names: Iterable[str] | None = None, *, dest: Path | None = None
) -> FinanceBenchPaths:
    """Download and verify the annotation files and the cited filings.

    Every file is checked against its pinned digest, including one already in
    the cache. Nothing is written outside the corpus cache.

    Parameters
    ----------
    doc_names : iterable of str or None, optional
        Filings to fetch. Defaults to all 84 the questions cite.
    dest : Path or None, optional
        Cache directory. Defaults to the corpus cache.

    Returns
    -------
    FinanceBenchPaths
        The verified files.

    Raises
    ------
    ChecksumMismatchError
        If any file differs from its pin.
    """
    root = cache_dir(dest)
    pdf_dir = root / "financebench-pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(PDF_SHA256 if doc_names is None else set(doc_names))
    return FinanceBenchPaths(
        open_source=fetch(FINANCEBENCH_OPEN_SOURCE, dest=root),
        document_information=fetch(FINANCEBENCH_DOCUMENT_INFORMATION, dest=root),
        pdfs={name: fetch(pdf_source(name), dest=pdf_dir) for name in names},
    )


__all__ = [
    "FINANCEBENCH_COMMIT",
    "FINANCEBENCH_DOCUMENT_INFORMATION",
    "FINANCEBENCH_LICENCE",
    "FINANCEBENCH_OPEN_SOURCE",
    "PAGE_BASE",
    "PDF_SHA256",
    "FinanceBenchPaths",
    "FinanceQuery",
    "fetch_financebench",
    "load_queries",
    "parse_queries",
    "pdf_source",
]
