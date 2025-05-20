import pandas as pd
from dataclasses import dataclass
import os

from util import read_excel_data

def combine_competency(competencies_df,competencies_dup_list):
    competencies_clean = competencies_df.copy()
    combine_description_dict = dict()

    for i in competencies_dup_list:
        selected_dup_df = competencies_df[competencies_df['competency'] == i]
        dup_list = list(selected_dup_df['description'])

        combine_text = ' '.join(dup_list)
        competencies = selected_dup_df['competency'].iloc[-1]
        print(f"Competencies: {competencies} and Combine text: {combine_text}")

        combine_description_dict[competencies] = combine_text

        competencies_clean = competencies_clean.drop(selected_dup_df.index,axis=0)

    combined_df = pd.DataFrame({
        'competency': list(combine_description_dict.keys()),
        'description': list(combine_description_dict.values())
    })

    clean_competencies_df = pd.concat([competencies_clean, combined_df], ignore_index=True)
    return clean_competencies_df

@dataclass
class DataTransformationConfig:
    competency:str=os.path.join("artifact",'competancy.csv')
    roles:str=os.path.join("artifact",'roles.csv')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.data_path = "EXCEL_PATH"
    def transforamtion_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        competencies_df,role_df = read_excel_data(self.data_path)
        
        print(f"Competency nul value :{competencies_df.isnull().sum()}")
        print(f"Roles null value: {role_df.isnull().sum()}")

        print(f"Competency duplicated value: {int(competencies_df.duplicated().sum())}")
        print(f"Roles duplicated value: {int(role_df.duplicated().sum())}")

        print(f"Length of competencies dataset: {len(competencies_df)}")
        print(f"Competenices course : {competencies_df['competency'].nunique()}")
        print(f"Length of roles dataset: {len(role_df)}")
        print(f"Roles positions : {role_df['role'].nunique()}")

        # Combine competency duplicated
        if len(competencies_df) != competencies_df['competency'].nunique():
            competencies_count_unique = competencies_df.groupby('competency')['description'].count().reset_index().sort_values(by='description', ascending=False)
            competencies_dup_list = list(competencies_count_unique[competencies_count_unique['description'] > 1]['competency'])
            clean_competencies_df = combine_competency(competencies_df,competencies_dup_list)

        os.makedirs(os.path.dirname(self.data_transformation_config.competency),exist_ok=True)
        clean_competencies_df.to_csv(self.data_transformation_config.competency,index=False,header=True)
        role_df.to_csv(self.data_transformation_config.roles,index=False,header=True)

        return clean_competencies_df,role_df
    
