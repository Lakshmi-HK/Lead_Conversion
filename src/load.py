import pandas as pd

path = "data/Leads.csv"

def load_data(path):
    df = pd.read_csv(path)
    return df

if __name__ ==  "__main__":
    df = load_data(path)
    print(df.info())