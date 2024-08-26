__import__('pysqlite3')
import argparse
import pandas as pd
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from tqdm.auto import tqdm
from haystack import Pipeline, Document
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from transformers import AutoTokenizer


def get_indexing_pipeline(dbname: str, model: str) -> Pipeline:
    pipe = Pipeline()
    ds = ChromaDocumentStore(persist_path=dbname)
    # add components to the pipeline
    device = ComponentDevice.from_str("cuda:0")
    pipe.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder(model=model, trust_remote_code=True, batch_size=8, device=device),
    )
    pipe.add_component("writer", DocumentWriter(ds))

    # connect the components
    pipe.connect("embedder", "writer")

    return pipe


def index_data(i_data, dbname: str, model: str):
    pipe = get_indexing_pipeline(dbname, model)
    res = pipe.run({"documents": i_data})
    return res


def df_to_haystack_documents(df: pd.DataFrame, id_key: str, text_key: str) -> list:
    documents = []
    for idx, row in tqdm(df.iterrows()):
        documents.append(
            Document(id=str(row[id_key]), content=row[text_key])
        )
    return documents


def count_max_tokens_df(df: pd.DataFrame, column: str, tokenizer: str) -> (int, int):
    max_tokens = 0
    overlen_cnt = 0
    t = AutoTokenizer.from_pretrained(tokenizer)
    print("Max tokens:", t.model_max_length)
    for idx, row in tqdm(df.iterrows()):
        try:
            to_text = t.encode(row[column])
            if len(to_text) > t.model_max_length:
                overlen_cnt += 1
            if len(to_text) > max_tokens:
                max_tokens = len(to_text)
        except:
            print(row)
    return max_tokens, overlen_cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Index data")
    parser.add_argument("--input_file", type=str, help="Path to the input jsonl file")
    parser.add_argument("--dbname", type=str, help="name of the output embedding store")
    parser.add_argument("--model", type=str, help="name of the sentence transformer model to use")

    args = parser.parse_args()
    data = pd.read_csv(args.input_file)
    data.dropna(inplace=True)
    print(data.head())
    documents = df_to_haystack_documents(data, id_key="hadm_id", text_key="other info:")
    # max_tokens, overlen_data_cnt = count_max_tokens_df(data, "other info:", args.model)
    # print(max_tokens, overlen_data_cnt)
    res = index_data(documents, args.dbname, args.model)
    print(res) 
