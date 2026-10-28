"""Test use case embed_chunks bằng đồ giả: SQLite in-memory, embedding giả, vector DB giả."""
import unittest
from typing import List, Optional

from src.application.use_cases.embed_chunks import EmbedChunksUseCase
from src.domain.models import EmbeddingVector, VectorRecord
from src.infrastructure.sql_database_adapters.models import Chunk
from src.infrastructure.sql_database_adapters.repositories import SqlDatabaseAdapter

DIM = 8


class FakeEmbedding:
    def embed_text(self, text):
        return EmbeddingVector(values=[0.1] * DIM, model_name="fake")

    def embed_batch(self, texts):
        return [EmbeddingVector(values=[float(len(t))] * DIM, model_name="fake", token_count=1) for t in texts]


class FakeVectorDb:
    def __init__(self, existing_dim: Optional[int] = None, fail_upsert: bool = False):
        self.collections = {} if existing_dim is None else {"c": existing_dim}
        self.fail_upsert = fail_upsert
        self.upserts: List[List[VectorRecord]] = []

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, name, dimension, metric=None):
        self.collections[name] = dimension

    def get_collection_dimension(self, name):
        return self.collections.get(name)

    def upsert(self, name, records):
        if self.fail_upsert:
            raise RuntimeError("qdrant down")
        self.upserts.append(list(records))


def create_test_setup(n_chunks: int, vector_db: Optional[FakeVectorDb] = None):
    sql = SqlDatabaseAdapter(database_url="sqlite:///:memory:")
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    sql._engine = engine
    sql._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    sql.chunk_repo._session_factory = sql._session_factory
    sql.embedding_repo._session_factory = sql._session_factory
    sql.init_db()
    with sql._session_factory() as s:
        for i in range(n_chunks):
            s.add(Chunk(chunk_id=f"c{i:03d}", text="x" * (i + 1), chunk_metadata={}, graph_extraction={}))
        s.commit()

    embedding_service = FakeEmbedding()
    use_case = EmbedChunksUseCase(
        sql_db=sql,
        embedding_service=embedding_service,
        vector_db=vector_db,
    )
    return use_case, sql, vector_db


class EmbedChunksUseCaseTest(unittest.TestCase):
    def test_sqlite_and_qdrant_receive_same_chunks_per_batch(self):
        vdb = FakeVectorDb()
        use_case, sql, _ = create_test_setup(5, vdb)
        stats = use_case.execute(
            provider="fake",
            model_name="fake",
            batch_size=2,
            dimension=DIM,
            push_qdrant=True,
            qdrant_collection="c",
        )
        self.assertEqual((stats.total, stats.successful, stats.failed), (5, 5, 0))
        self.assertEqual(sql.count_embeddings(), 5)
        batches = [[r.id for r in b] for b in vdb.upserts]
        self.assertEqual(sorted(sum(batches, [])), [f"c{i:03d}" for i in range(5)])  # không trùng, không thiếu
        self.assertEqual([len(b) for b in batches], [2, 2, 1])
        self.assertEqual(vdb.collections["c"], DIM)

    def test_second_run_embeds_nothing(self):
        vdb = FakeVectorDb()
        use_case, sql, _ = create_test_setup(3, vdb)
        use_case.execute(provider="fake", model_name="fake", dimension=DIM, push_qdrant=True, qdrant_collection="c")
        stats = use_case.execute(provider="fake", model_name="fake", dimension=DIM, push_qdrant=True, qdrant_collection="c")
        self.assertEqual(stats.total, 0)
        self.assertEqual(sql.count_embeddings(), 3)

    def test_limit(self):
        use_case, sql, _ = create_test_setup(5, None)
        stats = use_case.execute(provider="fake", model_name="fake", dimension=DIM, push_qdrant=False, limit=3)
        self.assertEqual(stats.successful, 3)
        self.assertEqual(sql.count_embeddings(), 3)

    def test_qdrant_failure_keeps_sqlite(self):
        vdb = FakeVectorDb(fail_upsert=True)
        use_case, sql, _ = create_test_setup(3, vdb)
        stats = use_case.execute(provider="fake", model_name="fake", dimension=DIM, push_qdrant=True, qdrant_collection="c")
        self.assertEqual(stats.successful, 3)
        self.assertEqual(sql.count_embeddings(), 3)
        self.assertEqual(sorted(stats.qdrant_failed_ids), ["c000", "c001", "c002"])

    def test_dimension_mismatch_raises_before_writing(self):
        vdb = FakeVectorDb(existing_dim=DIM + 1)
        use_case, sql, _ = create_test_setup(3, vdb)
        with self.assertRaises(ValueError):
            use_case.execute(provider="fake", model_name="fake", dimension=DIM, push_qdrant=True, qdrant_collection="c")
        self.assertEqual(sql.count_embeddings(), 0)
        self.assertEqual(vdb.upserts, [])


if __name__ == "__main__":
    unittest.main()
