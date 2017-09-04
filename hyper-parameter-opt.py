import numpy as np
import zsl_task as zsl
import csv

learning_rates = 10 ** np.random.uniform(-6, -2, size=15)
reg_const = 10 ** np.random.uniform(-6, -3, size = 15)
drop_outs = np.random.uniform(0.5,1,size = 15)
iterations = 200000

results_dir = "/home/huseyin/Work/Ml/Projects/zsl_task"

results = zsl.main(reg_const, learning_rates, iterations,drop_outs)

sorted_results = sorted(results,key= lambda x: x[2],reverse=True)

for res in sorted_results:
    print "On {0:05}. iteration, With lr: {1:.3E} , reg_const: {2:.3E}, keep_prop: {3:.5f} -> final va_acc: {4:0.4f}, final unregularized loss: {5:09.5f} "\
        .format(iterations,res[0],res[1],res[2],res[3],res[4])

with open(results_dir+"/results_multi_layer_non_linear.csv", 'a') as outcsv:

    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['iterations', 'learning_rate', 'regularization_const', "Keep_prop" ,"final va_acc", "final unregularized loss"])
    for res in sorted_results:

        #Write item to outcsv
        writer.writerow([iterations,res[0], res[1],res[2], res[3],res[4]])