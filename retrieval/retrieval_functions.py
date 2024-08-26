import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as pltimport 
from sentence_transformers import SentenceTransformer, util
import re

# init csv files into dataframe only include stay_id and the linking target
def init_icd_df(path): #col_name1 is in most cases the stay_id
    data = pd.read_csv(path)
    data = data[['stay_id', 'icd_code']]
    data = data.groupby('stay_id').agg(lambda x: sorted(x.tolist())) #group data with stay_id and sort the list
    return data
### !!! the list is in order in str format not in numeric format !!!

# level3 icd comparation
def compare_icd(given_df, given_stayid, target_df):
    temp_list = given_df.at[given_stayid, 'icd_code']
    print("given icd:",given_stayid,':',temp_list)
    Max_sim = 0
    simicd_list = []
    for icd_list in target_df['icd_code']:
        cur_sim = 0
        for icd in icd_list:
            if icd in temp_list:
                cur_sim += 1
        if cur_sim > Max_sim:
            Max_sim = cur_sim
            simicd_list = icd_list
    target_stayid = target_df[target_df['icd_code'].apply(tuple) == tuple(simicd_list)].index
    print("target icd:",target_stayid,':',simicd_list)
    return target_stayid

# level3 icd comparation with filter
def compare_icd_filtered(given_df, given_stayid, target_df):
    given_icdlist = given_df.at[given_stayid, 'icd_code']
    print("given icd:",given_stayid,':',given_icdlist)
    Max_sim = 0
    simicd_list = []
    for icd_list in target_df['icd_code']:
        cur_sim = 0
        for icd in icd_list:
            if icd in given_icdlist:
                cur_sim += 1
            elif icd not in given_icdlist:
                cur_sim = -1
                break
        if cur_sim > Max_sim:
            Max_sim = cur_sim
            simicd_list = icd_list
    target_stayid = target_df[target_df['icd_code'].apply(tuple) == tuple(simicd_list)].index
    print("target icd:",target_stayid,':',simicd_list)
    return target_stayid


# level 4 chiefcomplaint comparation

# init chiefcomplaint dataframe
def init_cc_df(path):
    data = pd.read_csv(path)
    data = data[['stay_id', 'chiefcomplaint']]
    data = data.groupby('stay_id').agg(lambda x: sorted(x.tolist()))
    data['chiefcomplaint'] = data['chiefcomplaint'].apply(lambda x: ' '.join(map(str, x)))
    data['chiefcomplaint'] = data['chiefcomplaint'].str.lower()
    return data

# level 4 jaccard similarity for chiefcomplaint
def jaccard_similarity(list1, list2):
    mlb = MultiLabelBinarizer()
    list1_encoded = mlb.fit_transform([list1, list2])[0]
    list2_encoded = mlb.transform([list2])[0]
    return jaccard_score(list1_encoded, list2_encoded)


def compare_jaccard(given_df, given_stayid, target_df):
    temp_list = given_df.at[given_stayid, 'chiefcomplaint']
    print(given_stayid,':',temp_list)
    similarities = target_df['chiefcomplaint'].apply(lambda x: jaccard_similarity(temp_list, x))
    target_stayid = similarities.idxmax()
    print(target_stayid,':',target_df.at[target_stayid, 'chiefcomplaint'])
    similarities = similarities[similarities != 0]
    similarities = similarities.sort_values(ascending=False)
    print(similarities)
    return target_stayid


# level 4 sentence transformer for chiefcomplaint
# added device argument to include the gpu in the computation
def init_st(df, device):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    df['embedding'] = None
    for index, row in df.iterrows():
        df.at[index, 'embedding'] = model.encode(row['chiefcomplaint'], convert_to_tensor=True).to(device)
    return df

def sentence_tr(embedding_1,embedding_2):
    return util.pytorch_cos_sim(embedding_1, embedding_2)

#this function returns the most similar stay_id
def compare_ST(given_df, given_stayid, target_df):
    target_df['similarity'] = None
    chiefcomplaint_list = given_df.at[given_stayid, 'chiefcomplaint']
    given_embedding = given_df.at[given_stayid, 'embedding']
    print("given chiefcomplaint:",given_stayid,':',chiefcomplaint_list,)
    for index, _ in target_df.iterrows():
        target_df.at[index, 'similarity'] = sentence_tr(given_embedding, target_df.at[index, 'embedding'])
    target_stayid = target_df['similarity'].idxmax()
    print("target chiefcomplaint:",target_stayid,':',target_df.at[target_stayid, 'chiefcomplaint'],',',target_df.at[target_stayid, 'similarity'])
    return target_stayid

#this function returns the target_df with similarities
def compare_ST_df(given_df, given_stayid, target_df):
    target_df['similarity'] = None
    chiefcomplaint_list = given_df.at[given_stayid, 'chiefcomplaint']
    given_embedding = given_df.at[given_stayid, 'embedding']
    print("given chiefcomplaint:",given_stayid,':',chiefcomplaint_list,)
    for index, _ in target_df.iterrows():
        target_df.at[index, 'similarity'] = sentence_tr(given_embedding, target_df.at[index, 'embedding'])
    target_df = target_df[['chiefcomplaint', 'similarity']]
    target_df = target_df.sort_values(by='similarity', ascending=False)
    return target_df


#level 5 sentence transformer for radiology

def rad_reader(text):
    keywords = ['examination:','history:', 'indication:', 'findings:', 'impression:', 'technique:', 'comparison:']
    pattern = '|'.join(map(re.escape, keywords))
    segments = re.split(f'({pattern})', text)
    result = {key: '' for key in keywords}
    current_key = None
    for segment in segments:
        if segment in keywords:
            current_key = segment
        elif current_key:
            result[current_key] += segment.strip()
    result = {key: value.strip() for key, value in result.items()}
    # remove all non-alphabetic characters
    result = {key: value for key, value in result.items() if re.search('[a-zA-Z]', value)}
    return result

def init_rad_df(path):
    data = pd.read_csv(path)
    data = data[['note_id','hadm_id', 'text']]
    data['text'] = data['text'].str.lower()
    #data = data[~data['text'].str.contains('comparison: none')]
    data['text'] = data['text'].apply(rad_reader)
    keywords = ['examination:','history:', 'indication:', 'findings:', 'impression:', 'technique:', 'comparison:']
    for keyword in keywords:
        data[keyword] = data['text'].apply(lambda x: x.get(keyword, ''))
    data = data[['note_id', 'hadm_id', 'examination:', 'history:', 'indication:', 'technique:', 'comparison:', 'findings:', 'impression:']]
    return data


#level 6 sentence transformer for discharge

def discharge_reader(text):
    keywords = ['name:','brief hospital course:','medications on admission:','discharge instructions:']
    pattern = '|'.join(map(re.escape, keywords))
    segments = re.split(f'({pattern})', text)
    result = {key: '' for key in keywords}
    current_key = None
    for segment in segments:
        if segment in keywords:
            current_key = segment
        elif current_key:
            result[current_key] += segment.strip()
    result = {key: value.strip() for key, value in result.items()}
    # remove all non-alphabetic characters
    result = {key: value for key, value in result.items() if re.search('[a-zA-Z]', value)}
    return result

def init_discharge_df(path):
    data = pd.read_csv(path)
    data = data[['note_id','hadm_id', 'text']]
    data['text'] = data['text'].str.lower()
    data['text'] = data['text'].apply(discharge_reader)
    keywords = ['name:','brief hospital course:','medications on admission:','discharge instructions:']
    for keyword in keywords:
        data[keyword] = data['text'].apply(lambda x: x.get(keyword, ''))
    data['name:'] = data['name:'].str.replace('\r', '')
    data['name:'] = data['name:'].str.replace('\n', '')
    data['name:'] = data['name:'].str.replace('"', '')
    data['medications on admission:'] = data['medications on admission:'].str.replace('\r', '')
    data['medications on admission:'] = data['medications on admission:'].str.replace('\n', '')
    data['medications on admission:'] = data['medications on admission:'].str.replace('"', '')
    data['brief hospital course:'] = data['brief hospital course:'].str.replace('\r', '')
    data['brief hospital course:'] = data['brief hospital course:'].str.replace('\n', '')
    data['brief hospital course:'] = data['brief hospital course:'].str.replace('"', '')
    data['discharge instructions:'] = data['discharge instructions:'].str.replace('\r', '')
    data['discharge instructions:'] = data['discharge instructions:'].str.replace('\n', '')
    data['discharge instructions:'] = data['discharge instructions:'].str.replace('"', '')
    data = data[['note_id', 'hadm_id', 'brief hospital course:', 'discharge instructions:','name:','medications on admission:']]
    return data

def extract_transitional_issues(row):
    transitional_issues_keyword = 'transitional issues:'
    brief_hospital_course_content = row['brief hospital course:']
    if transitional_issues_keyword in brief_hospital_course_content:
        parts = brief_hospital_course_content.split(transitional_issues_keyword, 1)
        row['brief hospital course:'] = parts[0].strip()
        row['transitional issues:'] = transitional_issues_keyword + parts[1].strip()
    else:
        row['transitional issues:'] = ''
    return row

# esstays transform for stay and hadm_id
def stay_to_hadm(stay_id, esstay_df):
    row = esstay_df[esstay_df['stay_id'] == stay_id]
    if not row.empty:
        hadm_id = row['hadm_id'].values[0]
        return stay_id, hadm_id
    else:
        return None

def hadm_to_stay(hadm_id, esstay_df):
    row = esstay_df[esstay_df['hadm_id'] == hadm_id]
    if not row.empty:
        stay_id = row['stay_id'].values[0]
        return stay_id, hadm_id
    else:
        return None


