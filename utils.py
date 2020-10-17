from dataPreparation import clean_text 
from nltk.tokenize import word_tokenize
import numpy as np


def merge_headlines(stock_frame,news_frame,
                    max_headline_length = 20,max_daily_length = 400):
    
    dates=[]
    price = []
    headlines = []
    all_headlines = []
   
    # For all the rows in the dataframe
    for row in stock_frame.iterrows():
        # define a new variable to store all the headlines for the day
        daily_headlines = []
        # Spot the date in the given row
        date = row[1]['Date']
        dates.append(date)
        # Store the price for the date
        #price.append(row[1]['cap'])
        
        for row_ in news_frame[news_frame.Date==date].iterrows():
            text = clean_text(row_[1]['News'])
            text_arr = word_tokenize(text)
            text_arr = text_arr[:max_headline_length]
            text_str = " ".join(text_arr)
            daily_headlines.append(text_str)
        
           
        if len(daily_headlines) > max_daily_length:  
            daily_news = daily_headlines[:max_daily_length]
            
        headlines.append(daily_headlines)            
    #Join all the news headlines in a day to to form a sentence 
   
    for head_line in headlines:
        daily_news = ' '
        for daily_headline in head_line:
            daily_news = daily_news + daily_headline + ' '        
        all_headlines.append(daily_news)
    
    return all_headlines,dates


def normalize_price(price):
    # Normalize opening prices (target values)
    max_price = max(price)
    min_price = min(price)
    mean_price = np.mean(price)
    def normalize(price):
        return ((price-min_price)/(max_price-min_price))
    
    norm_price = []
    for p in price:
        norm_price.append(normalize(p))
        
    return norm_price