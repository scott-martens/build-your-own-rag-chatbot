from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class JinaRerank(BaseDocumentCompressor):
    """Document compressor that uses `Jina Rerank API`."""

    client: Any = None
    """Jina client to use for compressing documents."""
    top_n: Optional[int] = 3
    """Number of documents to return."""
    model: str = "jina-reranker-v1-base-en"
    """Model to use for reranking."""
    jina_api_key: Optional[str] = None
    """Jina API key. Must be specified directly or via environment variable 
        JINA_API_KEY."""
    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        try:
            jina_api_key = convert_to_secret_str(
                get_from_dict_or_env(values, "jina_api_key", "JINA_API_KEY")
            )
        except ValueError as original_exc:
            try:
                jina_api_key = convert_to_secret_str(
                    get_from_dict_or_env(values, "jina_auth_token", "JINA_AUTH_TOKEN")
                )
            except ValueError:
                raise original_exc
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {jina_api_key.get_secret_value()}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )
        values["session"] = session
        return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        model: Optional[str] = None,
        top_n: Optional[int] = -1,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
        """  # noqa: E501
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = model or self.model
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        results = self._rerank(query, model, top_n, docs)
        result_dicts = []
        for res in results:
            result_dicts.append(
                {"index": res['index'], "relevance_score": res['relevance_score']}
            )
        return result_dicts

    def _rerank(self, query: str, model: str, top_n: int, docs: List[str]) -> List[Dict[str, Any]]:
        url = f"https://api.jina.ai/v1/rerank"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}"
        }

        data = {
            "model": model,
            "query": query,
            "documents": docs,
            "top_n": top_n
        }

        response = requests.post(url, headers=headers, json=data)
        js = response.json()
        if 'results' not in js:
            raise RuntimeError(js["detail"])
        return js['results']


    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Jina's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed