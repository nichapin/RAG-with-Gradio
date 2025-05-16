import pandas as pd


def read_excel_data(data_path):
    competencies_df = pd.read_excel(data_path,sheet_name="competencies")
    role_df = pd.read_excel(data_path,sheet_name="roles")
    return competencies_df,role_df
