import multiprocessing
from utils import thread_test_func

if __name__ == "__main__":
    # List of arguments for different runs
    arg_sets = [
        ('Process 1', 'banana', 5),
        ('Process 2', 'apple', 3),
        ('Process 3', 'orange', 1)
    ]

    with multiprocessing.Pool(processes=len(arg_sets)) as pool:
        results = pool.starmap(thread_test_func, arg_sets)

    print("All processes completed")
    for result in results:
        print(result)
