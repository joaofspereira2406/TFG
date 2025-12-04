import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_json("BBQ-AGE_predictions_2025-12-04_09-52-12.json", orient='records')

# Drop rows where label or prediction is None/NaN
df = df.dropna(subset=['label', 'prediction'])

acc = accuracy_score(df['label'], df['prediction'])
print("Accuracy:", acc)

from sklearn.metrics import confusion_matrix

mapping = {
    'Q1-OptionA': 'Option1',
    'Q1-OptionB': 'Option2',
    'Q1-OptionC': 'Option3',
    # Add all mappings for all questions
}

df['label_mapped'] = df['label'].map(mapping)
df['prediction_mapped'] = df['prediction'].map(mapping)

cm = confusion_matrix(df['label_mapped'], df['prediction_mapped'], labels=['Option1','Option2','Option3'])
print(cm)

