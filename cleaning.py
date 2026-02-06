import pandas as pd


def clean_Climate_Change_Dataset() -> pd.DataFrame:
    df = pd.read_csv("./Beginner_Climate_Change_Dataset_20_Features_1200_Rows.csv")
    df["country"] = df["country"].str.strip()
    df = pd.get_dummies(df, columns=["country"], drop_first=True)
    df = df.dropna()
    df = df.astype(float)
    return df


df = clean_Climate_Change_Dataset()
