"""workflow-RAG 的检索资产：语料切块、文本嵌入、本地向量库、检索器。

这个子包只在 ``--pipeline rag`` 时被用到，所以主流程不 import 它，
``chromadb`` / ``requests`` 之外的重依赖也不会拖累基础安装。
"""

from __future__ import annotations

__all__ = ["chunking", "embedding", "store", "retriever"]
