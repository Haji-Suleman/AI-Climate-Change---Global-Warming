import pandas as pd


def clean_Climate_Change_Dataset() -> pd.DataFrame:
    df = pd.read_csv("./Beginner_Climate_Change_Dataset_20_Features_1200_Rows.csv")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.dropna()
    return df
