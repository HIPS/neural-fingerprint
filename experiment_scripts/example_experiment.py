import os
import time

def output_file(filename):
    return os.path.join(os.environ["OUTPUT_DIR"], filename)

# Try chaning these parameters
x = 100
y = 51

print "This is a python script"
print "What is %s + %s?" % (x, y)
fout = output_file("output.txt")
print "Saving answer in %s" % fout
with open(fout, "w") as f:
    f.write("%s + %s = % s\n" % (x, y, x + y))
print "Countdown!"
for i in range(10, 0, -1):
    time.sleep(0.1)
    print i
