# BBQ Datasets for Fair-LLM-Benchmark

### Overview

This folder contains a collection of BBQ datasets used for evaluating biases in language models. Each dataset focuses on a specific demographic category, such as age, gender, race, or disability status.


The datasets were downloaded from the [Fair-LLM-Benchmark GitHub repository](https://github.com/i-gallegos/Fair-LLM-Benchmark)
and converted to JSON format for consistent processing.
Data last refreshed: 2025-11-18

### Datasets

|                                                        File | Description|
|------------------------------------------------------------:|-----------------:|
|                                                BBQ-AGE.json |	Scenarios and questions related to age bias |
|                                         BBQ-DISABILITY.json |	Scenarios and questions related to disability| 
|                                             BBQ-GENDER.json |	Scenarios and questions related to gender identity |
|                                        BBQ-NATIONALITY.json |	Scenarios and questions related to nationality |
|                                BBQ-PHYSICAL_APPEARANCE.json |	Scenarios related to physical appearance |
|                                     BBQ-RACE_ETHNICITY.json |	Scenarios and questions related to race and ethnicity|
|                                           BBQ-RACE_SES.json |	Scenarios related to race and socio-economic status|
|                                        BBQ-RACE_GENDER.json | Scenarios related to race and gender intersections|
|                                           BBQ-RELIGION.json |	Scenarios and questions related to religion|
|                                                BBQ-SES.json |	Scenarios related to socio-economic status|
|                                 BBQ-SEXUAL_ORIENTATION.json |	Scenarios related to sexual orientation|


### File Format

Format: JSON

Orientation: split (suitable for direct pandas loading)

Example usage in Python:
```python
import pandas as pd

df = pd.read_json("datasets/BBQ-AGE.json", orient="split")
print(df.head())

```

### Notes

Each dataset contains structured scenarios, questions, answer options, and correct answers.

No additional preprocessing has been applied beyond the original dataset conversion.

Ensure the `datasets` folder is in your working directory when loading files.

### Source & License

Original datasets: [Fair-LLM-Benchmark GitHub repository](https://github.com/i-gallegos/Fair-LLM-Benchmark)

License information: Please refer to the original repository.