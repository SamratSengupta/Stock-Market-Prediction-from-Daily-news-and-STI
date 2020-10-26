The current project is associated with the use of deep learning architectures to predict the intraday stock price changes of the Dow Jones Industrial Average (DJIA) with the help of daily news feed from Reddit and stock technical indicators.CNN-LSTM and LSTM models are used as baseline and Bert-capsule and tabnet are implmented as proposed models.Prediction weight based ensembling of proposed models are done with the help of a linear model to yield final output. The papers and presentation are submitted TO L.J.M.U 
The sequence project code execution is given below .

# Stock-Market-Prediction-from-Daily-news-and-STI

The codes are executed in folowing order

Steps 1,2 and 3 are optional as data has been already downloaded and merged

1> Manually download the Dowjones.csv and News.csv  from https://www.kaggle.com/aaron7sun/stocknews  (Date 8/8/2008 to 1/7/2016)
   Additional DJI.csv is also downloaded manually from yahoofinance site (https://in.finance.yahoo.com/quote/%5EDJI/history/) from (2/7/2016 to 1/9/2020)

2>Execute the Data-Download.ipynb to augment dowjones and reddit news files.The reddit news are downloaded yearly basis with pushshift api to form files under reddit-yearly-download. The news data collected are from 2016(from 2/7/2016),2017,2018,2019 and 2020(upto sep 20)

3>Execute the Data-Merge.ipynb to merge all the downloaded files and original file to dowjones_modified.csv and news_modified.csv as complete dataset for EDA.

These Steps (4 to 16)needs to be executed mandatorily to simulate the thesis work 

4>Manually download glove.840B.300d is downloaded manually from https://www.kaggle.com/takuok/glove840b300dtxt and used for word embedding in training CNN-LSTM model

5>Execute the EDA_Feature_derivation_DIJA.ipynb for conducting EDA on dowjones_modified.csv . It outputs  stock_original.csv (which contains original parameetrs) and stock_indicators.csv which contains Stock Technical Indicators derived in the same ipynb file.

6>Execute EDA_Data_preparation_RedditNews.ipynb for conducting EDA on news_modified.csv .The data cleaning methods are also derived from the comparison of raw and procesed data.

7>Run the EDA-data-process-split.ipynb to do data cleaning, data merging on news.csv and combining with dowjones.csv to form a dataframe which is split into 9:1 ratio (maintaining chronological order of date) one for model training as model.csv and other for performing ensemble as ensemble.csv

8>Run train-LSTM.ipynb to evaluate and tune hyperparameters for baseline LSTM model on model.csv (with STI data)

9>Run train-conv-lstm.ipynb to evaluate and tune hyperparameters for baseline CNN-LSTM model on model.csv (with news data)

10>Run train-tabnet.ipynb to evaluate and optimize hyperparameters for proposed Tabnet model on model.csv (with STI data)

-- pytorch's bert-base-uncased for tokenization is downloaded and also pre-trained bert-base-uncased is downloaded for modelling 
  
11>Run train-Bert-Capsule.ipynb to evaluate and tune hyperparameters for proposed Bert-Capsule model on model.csv (with news data)

-- After observing and evaluating all the models taking data from model.csv , observations are entered into model-evaluation.csv 
-- The best performing model with optimum hyperparameters are chosen for ensemble 

12>Alternatively download the trained models from  https://drive.google.com/u/1/uc?export=download&confirm=XFdQ&id=1iNSKZXWS8UoXGwcFMaIsK3_JzzERTLiD and keep in Models folder under the parent directory.

13>Execute ensemble-infer-valid.ipynb  to get inferred predicted change of price from bert-capsule model and tabnet model for data present in ensemble.csv to form
   ensemble-linear-features.csv contaning predictions from both models as well as actual change of price for a given date

13>Execute ensemble-feature-modeling.ipynb to produce a linear model on ensemble-linear-features.csv with predictions from model as predictors and chnage of price as target variable. the coefficients and intercept of the linear model is noted down for future use.

--the dowjones data is downloaded from Yahoo finance and reddit news is crawled from reddit for date Sept-2 to Sept-20 2020  to perform real time analysis of ensembled outcome

14>Execute real-time-prediction.ipynb  to produce final ensemble outcome on news_realtime.csv and dowjones_realtime.csv
