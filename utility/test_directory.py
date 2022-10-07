import os
import sys

print(os.getcwd())
print(os.path.abspath(__file__))
print(os.path.realpath(__file__))
print(sys.path[0])
print(sys.argv[0])


# set working directory
os.chdir(sys.path[0])
print(os.getcwd())