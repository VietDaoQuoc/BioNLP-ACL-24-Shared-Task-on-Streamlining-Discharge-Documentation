__import__('pysqlite3')
import argparse
import pandas as pd
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List

from pydash import chunk
from tqdm.auto import tqdm
from haystack import Pipeline, Document, component
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.utils import ComponentDevice, Device


@component
class ChromaRetriever:
    def __init__(self, db: ChromaDocumentStore, top_k: int = 3) -> None:
        self._db = db
        self.top_k = top_k

    @component.output_types(matches=dict)
    def run(self, documents: List[Document]):
        vectors = []
        for doc in documents:
            vectors.append(doc.embedding)

        matches = self._db._collection.query(
            query_embeddings=vectors, n_results=self.top_k, include=["documents", "distances"]
        )

        return {"matches": matches}


def get_query_pipeline(ds, model: str, top_k: int, device:str, batch_size: int) -> Pipeline:
    device = ComponentDevice.from_single(Device.cpu()) if device == "cpu" else ComponentDevice.from_single(Device.gpu(0))
    pipe = Pipeline()
    # add components to the pipeline
    pipe.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder(model=model, trust_remote_code=True, batch_size=batch_size, device=device),
    )
    pipe.add_component("retriever", ChromaRetriever(db=ds, top_k=top_k))

    # connect the components
    pipe.connect("embedder", "retriever")

    return pipe


def query(i_data, pipe: Pipeline):
    print(f"Querying {len(i_data)} documents...")
    res = pipe.run({"documents": i_data})
    return res


def df_to_haystack_documents(df: pd.DataFrame, id_key: str, text_key: str) -> list:
    documents = []
    for idx, row in tqdm(df.iterrows()):
        documents.append(
            Document(id=str(row[id_key]), content=row[text_key])
        )
    return documents


if __name__ == '__main__':
    # This is just a sample code snippet to demonstrate the query function
    parser = argparse.ArgumentParser(description="Query data")
    parser.add_argument("--input_file", type=str, help="Path to the input csv file")
    parser.add_argument("--dbname", type=str, help="name of the output embedding store")
    parser.add_argument("--model", type=str, help="name of the sentence transformer model to use")
    parser.add_argument("--device", type=str, default="cpu", help="device to use for embedding")
    parser.add_argument("--output_file", default='output/test_retrievedHadmID_tv_oneshot.csv', type=str,
                        help="Path to the output csv file")

    args = parser.parse_args()
    data = pd.read_csv(args.input_file)
    data.dropna(inplace=True)
    print(data.head())
    batch_size = 2

    # For dischage, id_key="hadm_id", text_key="other info:"
    documents = df_to_haystack_documents(data, id_key="hadm_id", text_key="other info:")
    # For radiology, id_key="hadm_id", text_key="findings_impression"
    # documents = df_to_haystack_documents(data, id_key="hadm_id", text_key="findings_impression")

    ds = ChromaDocumentStore(persist_path=args.dbname)
    pipeline = get_query_pipeline(ds, args.model, top_k=1, device=args.device, batch_size=batch_size)

    retrieved_hadm_id_dict = {}
    retrieved_hadm_id_dict['hadm_id'] = []
    retrieved_hadm_id_dict['retrieved_hadm_id'] = []

    doc_batches = chunk(documents, batch_size)
    for docs in tqdm(doc_batches):
        res = query(docs, pipeline)
        # print(res)
        for doc, ret_id in zip(docs, res['retriever']['matches']['ids']):
            retrieved_hadm_id_dict['hadm_id'].append(doc.id)
            retrieved_hadm_id_dict['retrieved_hadm_id'].append(ret_id[0])
    df = pd.DataFrame.from_dict(retrieved_hadm_id_dict)
    df.to_csv(args.output_file, index=False)