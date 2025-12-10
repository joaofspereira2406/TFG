import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


###qwen
df = pd.read_json("output_qwen.json", orient='records')
data = pd.json_normalize(df["data"].explode())

# Accuracy
acc = accuracy_score(data['label'], data['preds'])
print("Accuracy:", acc)

# Automatically extract and sort unique labels
labels = sorted(set(data['labels_polarity']) | set(data['preds_polarity']))
print("Detected labels:", labels)

# Confusion matrix (be sure order matches labels)
cm = confusion_matrix(data['labels_polarity'], data['preds_polarity'], labels=labels)
print(cm)

plt.figure(figsize=(10,7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues"
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


###gpt_neo
df = pd.read_json("output_gpt_neo.json", orient='records')
data = pd.json_normalize(df["data"].explode())

# Accuracy
acc = accuracy_score(data['label'], data['preds'])
print("Accuracy:", acc)

# Automatically extract and sort unique labels
labels = sorted(set(data['labels_polarity']) | set(data['preds_polarity']))
print("Detected labels:", labels)

# Confusion matrix (be sure order matches labels)
cm = confusion_matrix(data['labels_polarity'], data['preds_polarity'], labels=labels).T
print(cm)

plt.figure(figsize=(10,7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues"
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()