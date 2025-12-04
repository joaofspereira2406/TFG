import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_json("output.json", orient='records')

acc = accuracy_score(df['label'], df['preds'])
print("Accuracy:", acc)