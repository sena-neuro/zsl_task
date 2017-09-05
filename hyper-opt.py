import numpy as np
import zsl_task as zsl
import csv
import os


results_dir = os.getcwd()
learning_rates = 10 ** np.random.uniform(-6, -1, size=10)
reg_const = 10 ** np.random.uniform(-6, -3, size = 10)
iterations = 10000

# Results for experiments with hyper parameters
results = zsl.main(reg_const, learning_rates, iterations)

# Sort results based on validation accuracy
sorted_results = sorted(results,key= lambda x: x[2],reverse=True)

print "Results of conducted experiments sorted with validation accuracy"

for res in sorted_results:
    print "On {0:05}. iteration, With lr: {1:.3E} and reg_const: {2:.3E} -> final va_acc: {3:0.4f}, final unregularized loss: {4:09.5f} "\
        .format(iterations,res[0],res[1],res[2],res[3])

with open(results_dir+"/results.csv", 'a') as outcsv:

    # Configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['iterations', 'learning_rate', 'regularization_const', "final va_acc", "final unregularized loss"])
    for res in sorted_results:

        # Write item to outcsv
        writer.writerow([iterations,res[0], res[1], res[2],res[3]])
