import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np
import pandas as pd

def decontracted(phrase):
    if "'" in phrase:
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def clean_text(text):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''    
    # Convert words to lower case
    
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])    
    punctuations = set(string.punctuation.replace('.', ''))
    refined_text = ''.join([ch if ch not in punctuations else ' ' for ch in text ])  
    #tokens = [word for sent in nltk.sent_tokenize(refined_text) for word in nltk.word_tokenize(sent)]    
    
    text = refined_text.lower()    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        # Remove the contractions
        for word in text:
            new_text.append(decontracted(word))
        # Recreate the sentence
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    
    # Remove stop words
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)

    return text


def derive_technical_indicators(dj,n=7):
    
    def app_mean_close(x,row_num_inx):    
        x = x.tolist()
        row_count = int(x[row_num_inx])
        y=djmiss['Close'].rolling(window=row_count).mean().tolist()
        return y[row_count-1]

    def app_max_high(x,row_num_inx):    
        x = x.tolist()
        row_count = int(x[row_num_inx])
        y=djmiss['High'].rolling(window=row_count).max().tolist()
        return y[row_count-1]
    
    def app_min_low(x,row_num_inx):    
        x = x.tolist()
        row_count = int(x[row_num_inx])
        y=djmiss['Low'].rolling(window=row_count).min().tolist()
        return y[row_count-1]
    
    def calculate_stochaistic_D(row,n,row_num_inx):    
        row_num = row[row_num_inx]    
        if row_num >=n:
            df = dj[dj['row_num'].between(row_num-n, row_num)]
        else:
            df = dj[dj['row_num'].between(1, row_num)]        
        stock_d = df['stochaistic_k_percent'].sum() / n     
        return stock_d


    def calculate_momentum(row,n,row_num_inx,close_index):
        row_num = row[row_num_inx]
        curr_close = row[close_index]
        if row_num > n:
            prevclose_df = dj[dj['row_num']== (row_num-n)]
        else:
            prevclose_df = dj[dj['row_num'] == 1]    
        prev_close_val = float(prevclose_df['Close'].tolist()[0])    
        return curr_close - prev_close_val
    
    
    def calculate_roc(row,n,row_num_inx,close_index):    
        row_num = row[row_num_inx]    
        ct = row[close_index]    
        if row_num > n:
            prevclose_df = dj[dj['row_num']== (row_num-n)]
        else:
            prevclose_df = dj[dj['row_num'] == 1]    
        prev_close_val = float(prevclose_df['Close'].tolist()[0]) 
        
        ctn = ct - prev_close_val      
        return (ct/ctn) * 100
    
    
    def calculate_ad_oscillator(row,row_num_inx,close_index):    
        row_num = row[row_num_inx]    
        ct = row[close_index]    
        prevclose_df = dj[dj['row_num']== (row_num-1)]    
        if prevclose_df.empty:
            prevclose_df = dj[dj['row_num']== (row_num)]  
        prev_close_val = float(prevclose_df['Close'].tolist()[0])    
        retval = (row[high_index] - prev_close_val)/(row[high_index] - row[low_index])    
        return retval
    
    dj['MA'] = dj['Close'].rolling(window=7).mean()
    dj['HH']  = dj['High'].rolling(window=7).max()
    dj['LL']  = dj['Low'].rolling(window=7).min()
    
    djmiss = dj[dj['MA'].isna()]
    djmiss['row_num'] = np.arange(len(djmiss)) +1
    djmiss['row_num'] = djmiss['row_num'].astype(int)
    row_num_inx = djmiss.columns.get_loc("row_num")
    
    djmiss['HH'] = djmiss.apply(lambda x : app_max_high(x,row_num_inx),axis=1)
    djmiss['LL'] = djmiss.apply(lambda x : app_min_low(x,row_num_inx),axis=1)
    djmiss['MA'] = djmiss.apply(lambda x : app_mean_close(x,row_num_inx),axis=1)
    
    dj.update(djmiss)
    
    dj['row_num'] = np.arange(len(dj)) +1
    dj['row_num'] = dj['row_num'].astype(int)
    
    row_num_inx = dj.columns.get_loc("row_num")
    close_index = dj.columns.get_loc("Close")
    high_index  = dj.columns.get_loc("High")
    low_index  = dj.columns.get_loc("Low")
    
    dj['stochaistic_k_percent'] = ((dj['Close'] - dj['LL'])/(dj['HH'] - dj['LL'])) *100
    dj['stochaistic_D'] = dj.apply(lambda x : calculate_stochaistic_D(x,n,row_num_inx),axis=1)
    dj['Momentum'] = dj.apply(lambda x : calculate_momentum(x,n,row_num_inx,close_index),axis=1)
    dj['rate_of_change'] = dj.apply(lambda x : calculate_roc(x,n,row_num_inx,close_index),axis=1)
    dj['William_R_percent'] = (dj['HH'] - dj['Close'])/(dj['HH'] - dj['LL'])
    dj['AD_oscillator'] =  dj.apply(lambda x : calculate_ad_oscillator(x,row_num_inx,close_index),axis=1)
    dj['Disparity'] = (dj['Close']/dj['MA'])*100
    
    dj = dj[dj.Open.notnull()]
    djx = dj[dj['row_num']==1]
    roc_mean = dj[dj['row_num']!=1]['rate_of_change'].mean()
    djx['rate_of_change'] = roc_mean
    dj.update(djx)
    
    dj = dj.dropna()
    dj_org = dj[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close','cap']]
    dj_org.head(5)
    
    dj_ti = dj.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'MA', 'HH', 'LL','row_num'],axis=1)
    #dj_ti['Date'] = dj_ti.index
    
    #dj_ti = dj_ti.drop(['Date'],axis=1)
    
    return dj_ti
    
    

    
    
    
    