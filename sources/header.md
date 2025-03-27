 # Predict the MIC of compounds against pathogenic bacteria

Predictions are from an AI model trained on the wild-type accumulator subset of the [SPARK
dataset](https://doi.org/10.1021/acsinfecdis.8b00193), available to browse 
[here](https://huggingface.co/datasets/scbirlab/thomas-2018-spark-wt).

Predictions are given in micromolar (µM) and µg/mL. You can optionally have uncertainty scores calculated. 

This model was generated using [our Duvida framework](https://github.com/scbirlab/duvida), as a result of 
hyperparameter searches and selecting the model that performs best on unseen test data (from a scaffold split). 
Duvida also allows the calculation of uncertainty metrics based on training data.

Available species for prediction are:
- [_Klebsiella pneumoniae_](https://huggingface.co/scbirlab/spark-dv-fp-2503-kpn)

More to come in the next week!
