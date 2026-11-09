"""BM25 + character TF-IDF + reciprocal rank fusion on synthetic documents."""

from collections import Counter
import json
import math
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer


SPEC = {
    "title": "Hybrid Evidence RAG",
    "subtitle": "比较两条检索通道，用 RRF 融合排序并展示引用证据",
    "slug": "hybrid-evidence-rag",
    "accent": "#7c5cff",
    "sample_input": "P-101 泵振动升高时，为什么还要检查轴承润滑记录？",
    "input_label": "向合成工业资料提问",
}


def tokenize(text):
    """Retain Latin identifiers and Chinese bigrams without a segmenter."""
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(segment[index:index + 2] for index in range(len(segment) - 1))
    return tokens


def reciprocal_rank_fusion(rankings, constant=60):
    scores = Counter()
    for ranking in rankings:
        for rank, document_index in enumerate(dict.fromkeys(ranking), start=1):
            scores[document_index] += 1 / (constant + rank)
    return scores


class HybridRetriever:
    def __init__(self, documents):
        self.documents = documents
        texts = [document["title"] + " " + document["text"] for document in documents]
        self.term_counts = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / len(documents)
        self.document_frequency = Counter()
        for counts in self.term_counts:
            self.document_frequency.update(counts.keys())
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(texts)

    def bm25(self, query, k1=1.5, b=0.75):
        query_terms = set(tokenize(query))
        scores = []
        size = len(self.documents)
        for counts, length in zip(self.term_counts, self.lengths):
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                inverse_frequency = math.log(1 + (size - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1 - b + b * length / self.average_length)
                score += inverse_frequency * frequency * (k1 + 1) / denominator
            scores.append(score)
        return scores

    def search(self, query, top_k=3):
        if not query.strip():
            raise ValueError("请输入检索问题。")
        lexical = self.bm25(query)
        vector = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        lexical_ranking = sorted(
            (index for index, score in enumerate(lexical) if score > 0),
            key=lambda index: (-lexical[index], index),
        )
        vector_ranking = sorted(
            (index for index, score in enumerate(vector) if score >= 0.04),
            key=lambda index: (-vector[index], index),
        )
        # Weak character overlap alone should not manufacture an answer.
        if not lexical_ranking and (not len(vector) or max(vector) < 0.08):
            return []
        fused = reciprocal_rank_fusion([lexical_ranking, vector_ranking])
        ranking = sorted(fused, key=lambda index: (-fused[index], index))[:top_k]
        return [
            {
                **self.documents[index],
                "bm25": round(lexical[index], 4),
                "tfidf": round(float(vector[index]), 4),
                "rrf": round(fused[index], 6),
            }
            for index in ranking
        ]


def load_documents():
    path = Path(__file__).resolve().parent / "data" / "documents.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run(payload, client):
    question = str(payload.get("text", "")).strip()
    retriever = HybridRetriever(load_documents())
    evidence = retriever.search(question)
    if not evidence:
        return {
            "summary": "资料中没有找到足够的检索证据，请换一个有文档覆盖的问题。",
            "metrics": {"文档总数": len(retriever.documents), "召回片段": 0},
            "sections": [{"title": "证据不足", "text": "本地演示没有调用大模型。资料只包含少量合成教学片段。"}],
        }

    context = "\n\n".join(f"[{item['id']}] {item['title']}\n{item['text']}" for item in evidence)
    if client.enabled:
        answer = client.complete(
            "你是证据问答助手。只根据提供的片段回答，引用必须写成 [D01] 形式。"
            "区分文档内容与推测，证据不足就明确说明。所有文档都是合成教学资料。",
            f"问题\n{question}\n\n检索证据\n{context}",
            max_tokens=700,
        )
    else:
        answer = "本地检索已完成。以下为实际召回的证据摘录，供人工核验。\n\n" + "\n\n".join(
            f"[{item['id']}] {item['text']}" for item in evidence
        )

    return {
        "summary": answer,
        "metrics": {"文档总数": len(retriever.documents), "召回片段": len(evidence), "融合通道": 2},
        "sections": [
            {
                "title": "排序分解",
                "table": {
                    "columns": ["文档", "BM25", "TF-IDF 相似度", "RRF"],
                    "rows": [[item["id"], item["bm25"], item["tfidf"], item["rrf"]] for item in evidence],
                },
            },
            {"title": "证据来源", "items": [f"[{item['id']}] {item['title']} · {item['source']}" for item in evidence]},
            {"title": "算法说明", "text": "BM25 关注词项匹配，字符 TF-IDF 处理短语重合，RRF 对两个排序各累加 1/(60+名次)。分数用于排序，不是答案正确概率。"},
        ],
    }
