# ImmunoPredict

**An educational machine learning prototype exploring patterns associated with immunotherapy response.**

## Overview

ImmunoPredict is an interactive machine learning project investigating how clinical and biological variables may be associated with response to immune checkpoint blockade across different cancer types.

The project uses a logistic regression classification model trained on the Chowell research dataset.

## Features

- Interactive prediction interface built with Gradio
- Six input categories, including cancer type
- Probability-based classification output
- Research and methodology overview
- Model evaluation using a held-out test dataset

## Model Performance

| Metric | Test result |
|---|---:|
| Accuracy | 69.7% |
| ROC-AUC | 0.760 |
| Precision | 42.7% |
| Recall | 66.9% |
| F1 score | 0.521 |

The results should be interpreted in the context of the dataset and its limitations. Test accuracy was below the 75.3% majority-class baseline.

## Technology

- Python
- Pandas
- Scikit-learn
- Gradio
- Galaxy

## Limitations

ImmunoPredict is an educational research prototype, not a clinically validated prediction system. It must not be used for diagnosis, treatment selection or individual medical decisions.

## Research and Acknowledgements

ImmunoPredict was developed as a replication project inspired by the LORIS machine-learning framework and its application to predicting immunotherapy response.

The project focuses on reproducing a selected machine-learning workflow using the publicly available Chowell dataset. It uses logistic regression to investigate patterns associated with immune checkpoint blockade treatment outcomes.

The original research and LORIS framework provide the foundation for this work. The project adapts a selected part of that approach into an interactive educational prototype.

AI tools were used to assist with coding, implementation and development.


AI tools were used to assist with coding and development.

