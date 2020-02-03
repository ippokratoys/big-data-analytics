import time


def log_finish():
    elapsed_time = time.clock() - start_time
    print("Finished in {0:.2f} sec.".format(elapsed_time, "seconds"))
    return elapsed_time


def log_start(str="Staring task..."):
    global start_time
    print(str)
    start_time = time.clock()
