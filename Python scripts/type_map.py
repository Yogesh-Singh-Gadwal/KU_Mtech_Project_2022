import pandas as pd
import re
import time
from multiprocessing import Pool

# category analysis  for getting product, delivery and packaging.
def review_Category(i, review, rating):
    
    regex_delivery = r"\bdeliver"
    regex_package = r"\bpackag"
        
    if i % 10000 == 0:
        print("-- Working on {}th review --".format(i))
        
    matches_delivery = re.finditer(regex_delivery, review, re.MULTILINE | re.IGNORECASE)
    matches_package = re.finditer(regex_package, review, re.MULTILINE | re.IGNORECASE)
    
    if sum(1 for _ in matches_delivery):
        return 'delivery'
    elif sum(1 for _ in matches_package):
        return 'packaging'
    else:
        return 'product'

# Driver code
if __name__ == "__main__" :
    t1=time.time()
    df = pd.read_csv('output/Amazon_reviews_sentiment_scores_f1.csv')
    print(df.head(2))


    t2=time.time()
    with Pool() as pool:
        # prepare arguments
        items = list(df[["text","star_rating"]].itertuples(name=None))
        # execute tasks and process results in order
        result = pool.starmap(review_Category, items, chunksize=12)
    
    # process pool is closed automatically
    # print(f'Got result: {result}', flush=True)

    # with Pool(processes=10) as pool:


    print("pool took:",time.time()-t2)
    
    df.loc[:,'category_class'] = result
    
    print(df.head())
    
    df.to_csv('output/Amazon_reviews_Category_f1.csv', index=False)
    print("Complete process took:",time.time()-t1)
