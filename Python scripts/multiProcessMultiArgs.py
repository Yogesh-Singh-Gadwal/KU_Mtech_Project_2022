# example of using starmap() in the process pool with chunksize
from random import random
from time import sleep
from multiprocessing.pool import Pool
 
# task executed in a worker process
def task(identifier, value):
    # report a message
    print(f'Task {identifier} executing with {value}', flush=True)
    # block for a moment
    sleep(value)
    # return the generated value
    return (identifier , value)
 
# protect the entry point
if __name__ == '__main__':
    # create and configure the process pool
    with Pool(processes=10) as pool:
        # prepare arguments
        items = [(i, random()) for i in range(10)]
        print(items)
        # execute tasks and process results in order
        for result in pool.starmap(task, items, chunksize=10):
            print(f'Got result: {result}', flush=True)
    # process pool is closed automatically