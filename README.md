# MasterThesis

This project investigates heatwave propagation using spatiotemporal deep learning models. 
The goal is to predict and analyze the evolution of heatwaves using both binary labels 
(`is_heatwave`) and continuous Complex Network (CN) coefficients (e.g., clustering coefficient `CC`) based on the dataset from 1990-2010.

The pipeline includes:
- data preprocessing and alignment
- feature engineering using network coefficients (feature ablation)
- sequence generation for temporal modeling
- ConvLSTM-based training
- evaluation for both classification and regression tasks

- **data-exploration.ipynb**
  + initial data exploration, see distribution of the coefficients, data cleaning, fill missing values, inspect min/max values
  + align the is_heatwave dataset with other CN coefficients (CC, BC, DC, ID, OD)
  + decide on the data split and make yearly sequences for the model ConvLSTM
    
- **model1.ipynb**
  + creating a new dataset after cleaning the data in the data-exploration that is later used for experiments --> clean_complex_network_dataset.nc (not uploaded here due to its size)
  + check class imbalance
  + running ConvLSTM with target prediction is_heatwave and Clustering Coefficient (CC)
  + Model evaluation - binary classification (is_heatwave) and regression (CC)
  + Evaluation updated based on the best threshold
  + Visualisation of the predictions on a specific date
  + heatwave progression visualisation - day by day
  + Centroid of largest connected component - heatwave trajectory over average affected region
  + Did the predicted trajectory move through the actual heatwave region? - GIF
 
- **model-coefficients.ipynb**
This notebook contains an earlier attempt at building the pipeline.
  - Initial exploration of data and model setup
  - Testing different preprocessing strategies
  - Prototyping ConvLSTM training
  - Debugging normalization and alignment issues

  Differences from final version:
  - Less modular structure
  - Limited support for multiple targets
  - No unified evaluation pipeline
  - Incomplete handling of regression targets (e.g., scaling issues with ID/OD)

This notebook is kept for reference to document the development process and design decisions.
  
- **run-model-pipeline.ipynb**
This notebook contains the full end-to-end pipeline used for experiments.

  + **Data preparation**
  - Loading E-OBS and network coefficient datasets with spatial and temporal alignment 
  - Region selection (Iberia / Mediterranean Basin)

  + **Feature selection**
  - Flexible selection of input coefficients (`CC`, `BC`, `DC`, `ID`, `OD`, `is_heatwave`)
  - Support for ablation experiments 

  + **Target definition**
  - Binary classification: `is_heatwave`
  - Regression: `CC`, `OD`, etc.
  - Temporal shift (`t → t+1`) for prediction

  + **Sequence generation**
  - Creation of sliding windows per year
  - Input shape: `(N, seq_len, channels, lat, lon)`

  + **Normalization**
  - Per-channel normalization for input features
  - Target normalization for regression tasks

  + **Model**
  - ConvLSTM architecture
  - Configurable for classification and regression

  + **Training**
  - BCEWithLogitsLoss for binary task
  - MSE / SmoothL1 for regression
  - Class imbalance handling via `pos_weight`

  + **Evaluation**
  - Classification metrics: F1, IoU, Precision, Recall, PR-AUC, ROC-AUC
  - Regression metrics: RMSE, MAE, R², Pearson correlation

  + **Ablation experiments**
  - Systematic comparison of different input combinations
  - Results saved to CSV

  + **Visualization**
  - Heatmaps of predictions vs ground truth
  - Error maps
  - Spatial comparison of `is_heatwave` and network coefficients
  
- **bc_vs_hw.ipynb**
  + compare Betweenness Coefficients vs is_heatwave to see if BC is a relevant predictor
  
