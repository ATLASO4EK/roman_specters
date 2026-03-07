import joblib
import pandas as pd
import sys

kyes_brain = {
    "cerebellum":0,
    "cortex":1,
    "striatum":2
}
kyes_res = {
    0:"control",
    1:"endo",
    2:"exo"
}
def predict_to_keys(dct):
    keys_res = {
        0:"control",
        1:"endo",
        2:"exo"
    }
    new_dct = {}
    for keys, items in dct.items():
        new_dct[keys_res[keys]] = items
    return new_dct

def keys_b(value):
    return kyes_brain[value]

def NoiceRed(value):
    if value - 10000 < 0:
        return pd.NA
    else:
        return value - 10000

def preprocess_data(file):
    df = pd.read_csv(file, sep="\t")
    df = df.drop(columns=["#Wave", "Unnamed: 5", "#Intensity", "#X", "Unnamed: 1"])
    df = df.rename(columns={"#Y": "Roman shifts", "Unnamed: 3": "Counts"})
    df["Brain region"] = ((file.split("_")[0] + " ") * len(df)).strip().split(" ")
    df["Brain region"] = df["Brain region"].apply(keys_b)
    df["Counts"] = df["Counts"].apply(NoiceRed)
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index().drop(columns=["index"])

    return df

def predict(file):
    model = joblib.load('RandomForest(100)_NoiceRed.pkl')
    inputs = preprocess_data(file)
    predictions = model.predict(inputs)

    return predictions

if __name__ == "__main__":
    path = sys.argv[1]
    res = pd.Series(predict(path))
    values = predict_to_keys(res.value_counts().to_dict())
    print(values)
    print(next(key for key, value in values.items() if value == max(values.values())))

