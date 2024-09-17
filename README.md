# phm2012_domain_cnn_bi_lstm
Relevant code: CNN-Bi-LSTM model, domain adaptation model, relevant experimental data, and charts.  

The data sets used in this experiment (CWRU, PHM2012) have been uploaded to Google Cloud disk, and the data link is as follows: https://drive.google.com/drive/folders/1bXvElCVCr0smjgKW4a0feG5L8W9U08sq?usp=drive_link.\Because the file size is too large, the uploaded file must be decompressed.  

First, we use rul_pre_data.py code to preprocess the original data, which mainly includes: importing data, normalizing data, marking data with RUL label, and data slicing (expanding data volume). The rul_model.py code is the CNN-Bi-LSTM model that we built. The param_search.py code (which does a random search) determines the hyperparameters to be used for model training. The rul_train.py code indicates how the data is put into the model for training.  
  
Secondly, the rul_main.py code was used to import relevant data, train the model, and determine the hyperparameters related to the model training to train the model. After the training, the trained model was saved. The trained model is imported through the predict.py code and tested using the data set.Visualize the test results using rul_result_visualize.py code, and MAE and RMSE are recorded.  
  
Finally, we import the trained model through rul_domain code, freeze the CNN and Bi-LSTM layers, update the regression layer and add the domain adaptation layer to build the CNN-BI-LSTM model based on domain adaptation. The model hyperparameters remain unchanged, and in the domain_train.py code we determine how the model is trained. The trained model is saved, tested with predict.py code, visualized with rul_result_visualize.py code, and MAE and RMSE are recorded.  

We save all the trained models in the model folder.


