print("initialising...")
import retrieval_functions as rf
import pandas as pd
import sys
import query

__import__('pysqlite3')
import argparse
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from typing import List
from tqdm.auto import tqdm
from haystack import Pipeline, Document, component
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

##################### main #################
#input the stay_id and get the hadm_id
test_esstays_path = 'D:\\Semester_folder\\SS24\\project_seminar\\project_data\\src\\valid\\edstays.csv'                    #correct esstays path
test_esstaysdf = pd.read_csv(test_esstays_path)
input_stayid = input("please input a stay_id:")
input_stayid = int(input_stayid)
input_stayid, input_hadmid = rf.stay_to_hadm(input_stayid,test_esstaysdf)
if input_hadmid == None:
    print("stay_id not in context")
    sys.exit()

# use level 6 generate the df for the hadm_id and find the line for the hadm_id
test_discharge_path = 'D:\\Semester_folder\\SS24\\project_seminar\\github\\Project-Seminar-DSAI-DischargeMe\\Symptom similarity\\connection\\discharge_valid.csv'                #correct discharge path !!!already fixed discharge file
test_dischargedf = pd.read_csv(test_discharge_path)
input_discharge = test_dischargedf[test_dischargedf['hadm_id'] == input_hadmid]
#print('input_discharge:',input_discharge)

# use query find the best 3 match for the hadm_id output as dict
input_discharge.dropna(inplace=True)
print(input_discharge.head())
documents = query.df_to_haystack_documents(input_discharge)
ds = ChromaDocumentStore(persist_path='/nethome/vdquoc/project_dsai/semantic_search/radiology_emb.db')    #correct embedding path
pipeline = query.get_query_pipeline(ds, 'Snowflake/snowflake-arctic-embed-m-long', top_k=3)
#print(documents[0].content)
result_df = pd.DataFrame(columns=['given_hadm','res_hadm','distances'])
result_df['given_hadm'] = input_discharge['hadm_id'].copy()
for i in range(0,len(documents)):
    res = query([documents[i]], pipeline)
    ids = res['retriever']['matches']['ids'][0]
    distances = res['retriever']['matches']['distances'][0]
    result_df.at[i,'res_hadm'] = ids
    result_df.at[i,'distances'] = distances
print(result_df.head())

# find the stay_id for the elements in dict
