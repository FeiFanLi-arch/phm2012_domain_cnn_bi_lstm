# phm2012_domain_cnn_bi_lstm
Relevant code: basic model, domain adaptation model, relevant experimental data, and charts. 
Datas from https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website and https://www.femto-st.fr/en.  
First, we use rul_pre_data.py code to preprocess the original data, which mainly includes: importing data, normalizing data, marking data with RUL label, and data slicing (expanding data volume).  
The rul_model.py code is the CNN-Bi-LSTM model that we built.  
The param_search.py code (which does a random search) determines the hyperparameters to be used for model training.


