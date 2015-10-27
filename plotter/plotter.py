from pylab import plot, show, xlabel, ylabel, legend
import sys


prefix = "/Users/nlp/Documents/workspace/machine-learning-test/log/pos/"
names = ["lbfgs", "gdAll", "sgd", "gd500"]
for f in names:
    xs = []
    ys = []
    for line in open(prefix+f):
        if line.startswith("Iteration "):
            tokens = line.split()
            try:
                xs.append(float(tokens[-1][:-1]))
            except:
                xs.append(float(tokens[-2][:-1]))
            ys.append(float(tokens[2]))
    plot(xs, ys, label = f.split("/")[-1])
xlabel('Time (seconds)')
ylabel('Objective Function')
legend(loc = 4)
show()
