Forecasting stock prices has been a challenging problem, and it has attracted
many researchers in the areas of the economic market, financial analysis, and
computer science. The current study is associated with the use of deep learning
models to predict the intraday stock price changes of the Dow Jones Industrial
Average (DJIA) stock market index with the help of daily news feed from Reddit
and stock technical indicators. In recent years, Convolution neural network
(CNN), Recurrent neural network (RNN), and their variants are applied
successfully in this domain, the most prominent being a jointed Recurrent
convolution neural network (RCNN) architecture. Here, more advanced models
like Capsule net and Transformers (which have been proven very successful in
sentiment analysis) are studied for analyzing the impact of global events in form
of news. The premise of using these architectures is based on the fact that the
transformer Encoder extracts the deep semantic features of the news headlines
while capsule network captures the structural relationship of the texts to derive
meaningful features. Another relatively new architecture Tab-net which uses
sequential attention to decide feature importance is explored for processing stock
technical indicators. Not only Tab-net captures sequential features like Long
short-term memory (LSTM) but also uses attention mechanisms to deduce
insights from structured tabular data. Finally, an ensemble of Tab-net and Capsule
Transformer variants is carried out to project the overall impact of the daily news
feed as well as exchange Technical indicators on stock price change.

The thesis code execution is given in below order



# Stock-Market-Prediction-from-Daily-news-and-STI

The codes are executed in folowing order


Steps 1,2 and 3 are optional as data has been already downloaded and merged

1> Manually download the Dowjones.csv and News.csv is downloaded from https://www.kaggle.com/aaron7sun/stocknews manually (Date 8/8/2008 to 1/7/2016)
   and DJI.csv is also downloaded manually from yahoofinance site (https://in.finance.yahoo.com/quote/%5EDJI/history/) from (2/7/2016 to 1/9/2020)

2>Data-Download.ipynb to augment dowjones and reddit news files.The reddit news are downloaded yearly basis with pushshift api to form files under reddit-yearly-download.
  the news data collected are from 2016(from 2/7/2016),2017,2018,2019 and 2020(upto sep 20)

3>Data-Merge.ipynb merges all the downloaded files and original file to dowjones_modified.csv and news_modified.csv as complete dataset for EDA.


These Steps (4 to 16)needs to be executed mandatorily to simulate the thesis work 

4> manually download glove.840B.300d is downloaded manually from https://www.kaggle.com/takuok/glove840b300dtxt and used for word embedding in training CNN-LSTM model

5>EDA_Feature_derivation_DIJA.ipynb for conducting EDA on dowjones_modified.csv . It outputs  stock_original.csv (which contains original parameetrs) and stock_indicators.csv which contains Stock Technical Indicators derived in the same ipynb file.

6>EDA_Data_preparation_RedditNews.ipynb for conducting EDA on news_modified.csv .The data cleaning methods are also derived from the comparison of raw and procesed data.

7>EDA-data-process-split.ipynb is used to do data cleaning, data merging on news.csv and combining with dowjones.csv to form a dataframe which is split into 9:1 ratio (maintaining chronological order of date) one for model training as model.csv and other for performing ensemble as ensemble.csv

8>train-LSTM.ipynb to evaluate and tune hyperparameters for baseline LSTM model on model.csv (with STI data)


9>train-conv-lstm.ipynb to evaluate and tune hyperparameters for baseline CNN-LSTM model on model.csv (with news data)

10>train-tabnet.ipynb to evaluate and optimize hyperparameters for proposed Tabnet model on model.csv (with STI data)

-- pytorch's bert-base-uncased for tokenization is downloaded and also pre-trained bert-base-uncased is downloaded for modelling 
  
11>train-Bert-Capsule.ipynb to evaluate and tune hyperparameters for proposed Bert-Capsule model on model.csv (with news data)

-- After observing and evaluating all the models taking data from model.csv , observations are entered into model-evaluation.csv 
-- The best performing model with optimum hyperparameters are chosen for ensemble 

-- The trained models can be downloaded from https://drive.google.com/u/1/uc?export=download&confirm=XFdQ&id=1iNSKZXWS8UoXGwcFMaIsK3_JzzERTLiD and kept under Models folder in same directory as the code

12>ensemble-infer-valid.ipynb is run to get inferred predicted change of price from bert-capsule model and tabnet model for data present in ensemble.csv to form
   ensemble-linear-features.csv contaning predictions from both models as well as actual change of price for a given date

13>ensemble-feature-modeling.ipynb is run to produce a linear model on ensemble-linear-features.csv with predictions from model as predictors and chnage of price as target variable.
   the coefficients and intercept of the linear model is noted down for future use.

--the dowjones data is downloaded from Yahoo finance and reddit news is crawled from reddit for date Sept-2 to Sept-20 2020  to perform real time analysis of ensembled outcome

14>real-time-prediction.ipynb is finally run to produce final ensemble outcome on news_realtime.csv and dowjones_realtime.csv
