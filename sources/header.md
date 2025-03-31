 # Predict the MIC of compounds against pathogenic bacteria

Predictions are from an AI model trained on the wild-type accumulator subset of the [SPARK
dataset](https://doi.org/10.1021/acsinfecdis.8b00193), available to browse 
[here](https://huggingface.co/datasets/scbirlab/thomas-2018-spark-wt).

Predictions are given in micromolar (µM) and µg/mL. You can optionally have uncertainty scores calculated. These can take a few minutes, so please be patient.

This model was generated using [our Duvida framework](https://github.com/scbirlab/duvida), as a result of 
hyperparameter searches and selecting the model that performs best on unseen test data (from a scaffold split). 
Duvida also allows the calculation of uncertainty metrics based on training data.

Available species for prediction are:
- [_Acinetobacter baumannii_](https://huggingface.co/scbirlab/spark-dv-2503-abau)
- [_Brucella abortus_](https://huggingface.co/scbirlab/spark-dv-2503-babo)
- [_Escherichia coli_](https://huggingface.co/scbirlab/spark-dv-2503-ecol)
- [_Francisella tularensis_](https://huggingface.co/scbirlab/spark-dv-2503-ftul)
- [_Klebsiella pneumoniae_](https://huggingface.co/scbirlab/spark-dv-2503-kpne)
- [_Pseudomonas aeruginosa_](https://huggingface.co/scbirlab/spark-dv-2503-paer)
- [_Staphylococcus aureus_](https://huggingface.co/scbirlab/spark-dv-2503-saur)
- [_Streptococcus pneumoniae_](https://huggingface.co/scbirlab/spark-dv-2503-spne)
- [_Yersinia enterocolitica_](https://huggingface.co/scbirlab/spark-dv-2503-yent)
- [_Yersinia pestis_](https://huggingface.co/scbirlab/spark-dv-2503-ypes)

Click on the links above for training details, model configurations, and evaluation metrics.
