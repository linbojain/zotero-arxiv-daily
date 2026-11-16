"""PubMed retriever.

Retrieves articles that were added to PubMed (Entrez date) on the previous UTC day
and match a user-defined PubMed search query.

Config example (in CUSTOM_CONFIG):

source:
  pubmed:
    query: '"J Arthroplasty"[ta] OR arthroplasty[tiab]'
    max_results: 300      # optional, default 300
    days_back: 1          # optional, default 1 (= yesterday, UTC)
    api_key: ${oc.env:NCBI_API_KEY,null}   # optional
    email: null           # optional, recommended by NCBI
"""
from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Any, Optional
import xml.etree.ElementTree as ET

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubmedPaper(Paper):
    """Paper whose affiliations come from PubMed metadata instead of the LLM."""

    def generate_affiliations(self, openai_client, llm_params):
        return self.affiliations


def _cfg(retriever_config, key, default=None):
    value = retriever_config.get(key, default) if retriever_config is not None else default
    return default if value is None else value


def _request(url: str, params: dict, retry_num: int = 5, delay: int = 10) -> requests.Response:
    for i in range(retry_num):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response
        except Exception as e:
            if i == retry_num - 1:
                raise
            logger.warning(f"PubMed request failed: {e}. Retry in {delay} seconds.")
            sleep(delay)


def _text(elem: Optional[ET.Element]) -> str:
    return "".join(elem.itertext()).strip() if elem is not None else ""


@register_retriever("pubmed")
class PubmedRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)
        if self.retriever_config is None or not _cfg(self.retriever_config, "query"):
            raise ValueError("source.pubmed.query must be specified for pubmed")

    def _common_params(self) -> dict:
        params = {"tool": "zotero-arxiv-daily"}
        api_key = _cfg(self.retriever_config, "api_key")
        email = _cfg(self.retriever_config, "email")
        if api_key:
            params["api_key"] = api_key
        if email:
            params["email"] = email
        return params

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        query = _cfg(self.retriever_config, "query")
        max_results = int(_cfg(self.retriever_config, "max_results", 300))
        days_back = int(_cfg(self.retriever_config, "days_back", 1))
        if self.config.executor.debug:
            max_results = min(max_results, 10)
            days_back = max(days_back, 7)  # make sure the test run finds something

        today = datetime.now(timezone.utc).date()
        min_date = (today - timedelta(days=days_back)).strftime("%Y/%m/%d")
        max_date = (today - timedelta(days=1)).strftime("%Y/%m/%d")
        logger.info(f"PubMed query: {query} | Entrez date {min_date} - {max_date}")

        search = _request(f"{EUTILS}/esearch.fcgi", {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "datetype": "edat",
            "mindate": min_date,
            "maxdate": max_date,
            "retmax": max_results,
            "retmode": "json",
            "sort": "pub_date",
        }).json()["esearchresult"]
        pmids = search.get("idlist", [])
        logger.info(f"PubMed found {search.get('count', 0)} articles, fetching {len(pmids)}")
        if not pmids:
            return []

        papers = []
        for start in range(0, len(pmids), 100):
            batch = pmids[start:start + 100]
            sleep(0.5)  # NCBI allows 3 requests/s without an API key
            xml_text = _request(f"{EUTILS}/efetch.fcgi", {
                **self._common_params(),
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            }).text
            papers.extend(self._parse_xml(xml_text))
        return papers

    @staticmethod
    def _parse_xml(xml_text: str) -> list[dict[str, Any]]:
        root = ET.fromstring(xml_text)
        results = []
        for article in root.findall(".//PubmedArticle"):
            citation = article.find("MedlineCitation")
            if citation is None:
                continue
            pmid = _text(citation.find("PMID"))
            art = citation.find("Article")
            if art is None:
                continue
            title = _text(art.find("ArticleTitle"))

            parts = []
            for abs_text in art.findall("Abstract/AbstractText"):
                label = abs_text.get("Label")
                content = _text(abs_text)
                if content:
                    parts.append(f"{label}: {content}" if label else content)
            abstract = "\n".join(parts)

            authors, affiliations = [], []
            for author in art.findall("AuthorList/Author"):
                collective = _text(author.find("CollectiveName"))
                name = collective or " ".join(
                    x for x in [_text(author.find("ForeName")), _text(author.find("LastName"))] if x
                )
                if name:
                    authors.append(name)
                for aff in author.findall("AffiliationInfo/Affiliation"):
                    aff_text = _text(aff)
                    if aff_text and aff_text not in affiliations:
                        affiliations.append(aff_text)

            journal = _text(art.find("Journal/ISOAbbreviation")) or _text(art.find("Journal/Title"))

            ids = {i.get("IdType"): _text(i) for i in article.findall("PubmedData/ArticleIdList/ArticleId")}
            results.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "affiliations": affiliations,
                "journal": journal,
                "doi": ids.get("doi"),
                "pmc": ids.get("pmc"),
            })
        return results

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        if not raw_paper["abstract"]:
            return None  # cannot be ranked or summarized without an abstract
        url = f"https://pubmed.ncbi.nlm.nih.gov/{raw_paper['pmid']}/"
        if raw_paper.get("pmc"):
            link = f"https://pmc.ncbi.nlm.nih.gov/articles/{raw_paper['pmc']}/"
        elif raw_paper.get("doi"):
            link = f"https://doi.org/{raw_paper['doi']}"
        else:
            link = url
        title = raw_paper["title"]
        if raw_paper.get("journal"):
            title = f"{title} [{raw_paper['journal']}]"
        # Keep only the institution part of each affiliation to stay short.
        affiliations = []
        keywords = ("universit", "hospital", "institut", "college", "school", "center", "centre", "clinic")
        for aff in raw_paper["affiliations"]:
            segments = [s.strip() for s in aff.split(",") if s.strip()]
            short = next((s for s in segments if any(k in s.lower() for k in keywords)),
                         segments[0] if segments else aff)
            if short not in affiliations:
                affiliations.append(short)
        return PubmedPaper(
            source=self.name,
            title=title,
            authors=raw_paper["authors"],
            abstract=raw_paper["abstract"],
            url=url,
            pdf_url=link,
            full_text=None,
            affiliations=affiliations or None,
        )
