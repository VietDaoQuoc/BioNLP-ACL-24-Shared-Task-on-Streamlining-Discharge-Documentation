from litellm import completion, encode
import os
import pandas as pd
import re
from evaluate import load
from tqdm.auto import tqdm
from litellm.litellm.utils import TextCompletionResponse

import time
start_time = time.time()

def litellm_generate_call(prompt) -> TextCompletionResponse:

    os.environ['ANYSCALE_API_KEY'] = "........"
    response = completion(
        model="anyscale/meta-llama/Meta-Llama-3-70B-Instruct", 
        messages=prompt, temperature=0.8, top_p=0.9, stream=False, max_tokens=1000
    )
    return response

def load_retrieved_hadm_id() -> pd.DataFrame:
    test_retrieved_hadm_id = pd.read_csv("/nethome/vdquoc/project_dsai/viet/hugging_face/test_retrievedHadmID_tv_oneshot.csv")

    return test_retrieved_hadm_id

def load_train_valid_discharge() -> pd.DataFrame:
    discharge_val = pd.read_csv("/nethome/vdquoc/project_dsai/data/valid/discharge.csv")
    print('valid shape', discharge_val.shape)
    print(discharge_val.columns)

    discharge_train = pd.read_csv("/nethome/vdquoc/project_dsai/data/train/discharge.csv")
    print('train shape', discharge_train.shape)
    print(discharge_train.columns)

    data = pd.merge(discharge_val, discharge_train, how='outer', on=['note_id', 'subject_id', 'hadm_id', 'note_type', 'note_seq', 'charttime', 'storetime', 'text'])

    data.rename(columns = {'text':'discharge_summary'}, inplace = True)
    data['discharge_instructions'] = data['discharge_summary'].apply(lambda x: re.findall(r'Discharge Instructions:\n(.*?)Followup Instruction', x, re.DOTALL))
    data['brief_hospital_course'] = data['discharge_summary'].apply(lambda x: re.findall(r'Brief Hospital Course:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', x, re.DOTALL))
    
    data['json'] = data.apply(lambda x: x.to_json(), axis=1)

    return data

def load_test_discharge() -> pd.DataFrame:
    ###Load and remove certain part from the discharge_summary of TEST AND VALID datasets

    discharge_test = pd.read_csv("/nethome/vdquoc/project_dsai/data/test_phase_2/discharge.csv")
    discharge_test.rename(columns = {'text':'discharge_summary'}, inplace = True)
    discharge_test['discharge_summary'] = [re.sub(r'Discharge Instructions:\n(.*?)Followup Instruction', '', x, flags=re.DOTALL) for x in discharge_test['discharge_summary']]
    discharge_test['discharge_summary'] = [re.sub(r'Brief Hospital Course:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', '', x, flags=re.DOTALL) for x in discharge_test['discharge_summary']]
    discharge_test['history_present_illness'] = discharge_test['discharge_summary'].apply(lambda x: re.findall(r'History of Present Illness:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', x, re.DOTALL))
    discharge_test['past_medical_history'] = discharge_test['discharge_summary'].apply(lambda x: re.findall(r'Past Medical History:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', x, re.DOTALL))
    discharge_test['discharge_medication'] = discharge_test['discharge_summary'].apply(lambda x: re.findall(r'Discharge Medications:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', x, re.DOTALL))
    discharge_test['json'] = discharge_test.apply(lambda x: x.to_json(), axis=1)
    
    return discharge_test

def generate_response_test(test_sample, test_retrieved_hadm_id_tv, train_valid_discharge) -> dict:

    test_phase_2_discharge_target = pd.read_csv("/nethome/vdquoc/project_dsai/data/test_phase_2/discharge_target.csv")

    ##### Filtered hadm_id for oneshot prompting with modification ######
    filtered_hadm_id_1_turn = pd.read_csv("/nethome/vdquoc/project_dsai/output_files/filtered_hadm_ID_for_1_turn.csv")

    dict_to_csv = {}
    dict_to_csv['hadm_id'] = []
    dict_to_csv['expected_discharge_instruction'] = []
    dict_to_csv['generated_discharge_instruction'] = []

    dict_to_csv['expected_brief_hospital_course'] = []
    dict_to_csv['generated_brief_hospital_course'] = []

    for index, row in tqdm(test_sample.iterrows()):
        hadm_id = row['hadm_id']

        retrieved_hadm_id = test_retrieved_hadm_id_tv.loc[hadm_id == test_retrieved_hadm_id_tv['hadm_id'], 'retrieved_hadm_id'].iloc[0]
        sample_input_data = train_valid_discharge.loc[train_valid_discharge['hadm_id'] == retrieved_hadm_id]

        sample_output_di = sample_input_data['discharge_instructions']
        sample_output_bhc = sample_input_data['brief_hospital_course']
        del sample_input_data['discharge_instructions']
        del sample_input_data['brief_hospital_course']
        del sample_input_data['json']

        sample_input_data.loc[:, 'json'] = sample_input_data.apply(lambda x: x.to_json(), axis=1)

        sample_input = str(sample_input_data['json'])

        query_input = str(row['json'])


        if (filtered_hadm_id_1_turn['hadm_id'].isin([hadm_id]).any()):
            print(hadm_id)
            prompt_di = f'''
                ###Instructions###
                Generate `DISCHARGE INSTRUCTION` for the patient based on the following data:{query_input}. \n

                The `DISCHARGE INSTRUCTION` must provide actionable information to help them take care of themselves when they leave the hospital. \n

                ###Example###
                The following is just a sample data, containing an `EXAMPLE_DISCHARGE INSTRUCTION` for the `EXAMPLE_INPUT`. \n `EXAMPLE_INPUT`: {sample_input} `EXAMPLE_DISCHARGE INSTRUCTION`: {sample_output_di} \n
            

                ###Content###
                The `DISCHARGE INSTRUCTION` of each patient of ‘subject_id’ and ‘hadm_id’ should be an overview of the conditions of the corresponding patient. \n

                The content of `DISCHARGE INSTRUCTION` must contain information from ‘discharge_summary’.\n

                Look into ‘history_present_illness’, ‘past_medical_history’ and ‘discharge_medication’ sentence by sentence to extract information on diagnosis and treatments administered. \n

                Avoid using any place holders. \n

                Avoid using '[insert diagnosis(s) e.g., pneumonia, heart failure]', instead provide specific diagnoses and treatments. \n

                The `DISCHARGE INSTRUCTION` must be written in simple terms. \n

                Provide information about medical jargon \n

                Start your response with: `Dear Ms. /Mr` followed by `____`, which is based on the ‘gender’ from the `EXAMPLE_INPUT`. \n

                Give an answer to a given question in a natural, human-like manner. \n

                Avoid adding any personal information about the patient such as name and date of birth. \n
                '''

            prompt_bhc = f'''
            ###Instructions###
            Generate `BRIEF HOSPITAL COURSE` for the patient based on the following data:{query_input}. \n
    
            The `BRIEF HOSPITAL COURSE` must be brief overview of what happened during a patient's stay in the hospital. \n

            ###Example###
            The following is just a sample data, containing an `EXAMPLE_BRIEF HOSPITAL COURSE` for the `EXAMPLE_INPUT`. \n `EXAMPLE_INPUT`: {sample_input} `EXAMPLE_BRIEF HOSPITAL COURSE`: {sample_output_bhc} \n
            
            ###Content###
            The `BRIEF HOSPITAL COURSE` of each patient of ‘subject_id’ and ‘hadm_id’ must be brief overview of the treatment provided and patient's response during a patient's stay in the hospital. \n

            The content of `BRIEF HOSPITAL COURSE` must contain details about the treatments or procedures the patient underwent, how the patient responded to these treatments, and any significant changes in the patient's health condition.\n
            
            Look into ‘history_present_illness’ and ‘past_medical_history’ to extract information on diagnosis, treatments administered, and discharge medications. \n
            
            The `BRIEF HOSPITAL COURSE` must be written in medical terms so that doctors can understand about the overall care of a patient. \n
            
            Avoid using any place holders. \n
            
            Avoid using '[insert diagnosis(s) e.g., pneumonia, heart failure]', instead provide specific diagnoses and treatments. \n
            
            Provide information about medical jargon. \n
            
            Start your response with: `Mr/Ms. ____ was` , which is based on the ‘gender’ from the `EXAMPLE_INPUT`. \n
    
            Avoid adding any personal information about the patient such as name and date of birth. \n
            '''

            print('--------------------------')
            messages_di = [
                { "content":prompt_di,"role": "user"},
            ]
            messages_bhc = [
                { "content":prompt_bhc,"role": "user"},
            ]

            response_di = litellm_generate_call(messages_di)
            response_bhc = litellm_generate_call(messages_bhc)
            print(response_di['choices'][0]['message']['content'])

            correcting_prompt_di = f'''
                ###Instructions###
                Generate a `CORRECTED DISCHARGE INSTRUCTION` with symptoms for the patient based on the `GENERATED DISCHARGE INSTRUCTION` {response_di['choices'][0]['message']['content']} of the following data:{query_input}. \n

                The `CORRECTED DISCHARGE INSTRUCTION` is a information-corrected summary of medical condition of the patient. \n

                ###Example###
                The information about the patient's medical condition in the `GENERATED DISCHARGE INSTRUCTION` {response_di['choices'][0]['message']['content']} may not be correct.\n

                ###Content###
                The `CORRECTED DISCHARGE INSTRUCTION` of each `GENERATED DISCHARGE INSTRUCTION` should be an overview of the conditions of the corresponding patient. \n

                Look into 'History of Present Illness': 'history_present_illness' and 'Past Medical History': 'past_medical_history' to correct the may false informations on on diagnosis, treatments administered, and discharge medications of the `GENERATED DISCHARGE INSTRUCTION. \n

                The `GENERATED DISCHARGE INSTRUCTION` may contain 'Discharge Medications' and is sometimes not needed in the for 'DISCHARGE INSTRUCTION'. \n
                Check again the 'sample_input' to see if the 'sample_output' has 'Discharge Medications'. \n

                Avoid using any place holders. \n

                Avoid using '[insert diagnosis(s) e.g., pneumonia, heart failure]', instead provide specific diagnoses and treatments. \n

                '''

            correcting_prompt_bhc = f'''
                ###Instructions###
                Generate a `CORRECTED BRIEF HOSPITAL COURSE` with symptoms for the patient based on the `GENERATED BRIEF HOSPITAL COURSE` {response_bhc['choices'][0]['message']['content']} of the following data:{query_input}. \n

                The `CORRECTED BRIEF HOSPITAL COURSE` is a brief overview of what happened during a patient's stay in the hospital. \n

                ###Example###
                The information about the patient's medical condition in the `GENERATED BRIEF HOSPITAL COURSE` {response_bhc['choices'][0]['message']['content']} may not be correct.\n

                ###Content###
                The `CORRECTED BRIEF HOSPITAL COURSE` of each `GENERATED BRIEF HOSPITAL COURSE` should be an overview of the conditions of the corresponding patient. \n

                The `CORRECTED BRIEF HOSPITAL COURSE` must be written in medical terms so that doctors can understand about the overall care of a patient. \n

                Look into 'History of Present Illness': 'history_present_illness' and 'Past Medical History': 'past_medical_history' to correct the may false informations on on diagnosis, treatments administered, and discharge medications of the `GENERATED DISCHARGE INSTRUCTION. \n

                Avoid using any place holders. \n

                Avoid using '[insert diagnosis(s) e.g., pneumonia, heart failure]', instead provide specific diagnoses and treatments. \n

                '''
            
            correcting_message_bhc = [{ "content": correcting_prompt_bhc, "role": "user"}]
            corrected_response_bhc = litellm_generate_call(correcting_message_bhc)

            correcting_message_di = [{ "content": correcting_prompt_di, "role": "user"}]
            corrected_response_di = litellm_generate_call(correcting_message_di)

            dict_to_csv['hadm_id'].append(hadm_id)
            dict_to_csv['expected_discharge_instruction'].append(test_phase_2_discharge_target.loc[test_phase_2_discharge_target['hadm_id'] == row['hadm_id'], 'discharge_instructions'].iloc[0])
            dict_to_csv['generated_discharge_instruction'].append(corrected_response_di['choices'][0]['message']['content'])

            dict_to_csv['expected_brief_hospital_course'].append(test_phase_2_discharge_target.loc[test_phase_2_discharge_target['hadm_id'] == row['hadm_id'], 'brief_hospital_course'].iloc[0])
            dict_to_csv['generated_brief_hospital_course'].append(corrected_response_bhc['choices'][0]['message']['content'])

    return dict_to_csv

if __name__ == '__main__':

    train_valid_discharge = load_train_valid_discharge()

    test_discharge = load_test_discharge()

    retrieved_hadm_id = load_retrieved_hadm_id()

    test_sample = test_discharge.sample(n=500, random_state=42)
    
    output = generate_response_test(test_sample=test_sample, test_retrieved_hadm_id_tv=retrieved_hadm_id, train_valid_discharge=train_valid_discharge)
    df = pd.DataFrame.from_dict(output)
    df.to_csv('/nethome/vdquoc/project_dsai/output_files/output_one_shot_with1turn_test_tv.csv')
    print('time of version oneshot_with_1_turn on TEST_PHASE_2 with train_valid combined')

    print("--- %s seconds ---" % (time.time() - start_time))