# MasterThesis

Focus on exploring the Complex Network (CN) coefficients based on the dataset from 1990-2010

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
- **run-model-pipeline.ipynb**
- **bc_vs_hw.ipynb** (compare Betweenness Coefficients vs is_heatwave to see if BC is a relevant predictor)
  
