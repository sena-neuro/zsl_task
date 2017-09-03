import numpy as np
import zsl_task as zsl

learning_rates = 10 ** np.random.uniform(-6, -1, size=10)
reg_const = 10 ** np.random.uniform(-6, -3, size = 10)
iterations = 1000 # for fast calculation may change later


# Results for experiments with hyper parameters
results = zsl.main(reg_const, learning_rates, iterations)

# Sort results based on validation accuracy
sorted_results = sorted(results,key= lambda x: x[2],reverse=True)

for res in sorted_results:
    print "On {0:05}. iteration, With lr: {1:.3E} and reg_const: {2:.3E} -> final va_acc: {3:0.4f}, final unregularized loss: {4:09.5f} "\
        .format(iterations,res[0],res[1],res[2],res[3])

