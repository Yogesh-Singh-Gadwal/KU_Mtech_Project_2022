import pandas as pd
import time
from multiprocessing import Pool
# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create a SentimentIntensityAnalyzer object.
sentiment_obj = SentimentIntensityAnalyzer()

# function to caliculate sentiments of the sentence.
def sentiment_scores(i,sentence, rating):
    global sentiment_obj
    
    if i % 1000 == 0:
        print("-- Working on {}th review --".format(i))

	# polarity_scores method of SentimentIntensityAnalyzer
	# object gives a sentiment dictionary.
	# which contains pos, neg, neu, and compound scores.
    sentiment_dict = sentiment_obj.polarity_scores(sentence)
    com =sentiment_dict['compound']
    sentiment_dict['compound']=0
    Keymax = max(sentiment_dict, key= lambda x: sentiment_dict[x])

	# decide sentiment as positive, negative and neutral
    if Keymax == "pos" or com >= 0.05:
        if rating == 4 or rating == 5:
            return "Positive"
        else :
            return "Fake"
        
    elif Keymax == "neg" or com <= - 0.05:
        if rating == 1 or rating == 2:
            return "Negative"
        else :
            return "Fake"

    elif Keymax == "neu" or (com > - 0.05 and com < 0.05):
        if rating == 3 or rating == 4 or rating == 2:
            return "Neutral"
        else :
            return "Fake"


# Driver code
if __name__ == "__main__" :
    
    df = pd.read_csv('output/Amazon_reviews_cleaned_f2.csv')
    print(df.head(2))

	# function calling
    # sentence = [0,"The plot was good, but the characters are uncompelling and the dialog is not great.",4]
    # print(sentiment_scores(sentence))

    t1=time.time()
    with Pool() as pool:
        # prepare arguments
        items = list(df[["text","star_rating"]].itertuples(name=None))
        # execute tasks and process results in order
        result = pool.starmap(sentiment_scores, items, chunksize=12)
    
    # process pool is closed automatically
    # print(f'Got result: {result}', flush=True)
    # create and configure the process pool
    # with Pool(processes=10) as pool:
    # execute tasks in order
        # list_res= pool.map(sentiment_scores, df['review'], chunksize=744940)
    # process pool is closed automatically

    print("pool took:",time.time()-t1)
    
    df.loc[:,'sentiment_class'] = result
    print(df.head())
    df.to_csv('output/Amazon_reviews_sentiment_scores_f1.csv', index=False)
