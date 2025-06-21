import os
import sys

from .executables import *

args = sys.argv[1:]
try:
    command = args[0]
except:
    print("Unknown error")
    sys.exit(-1)

if (command == "reduce"):
    wiyn_benchpipe(args[1:])
elif (command == "region"):
    region_file_from_output(args[1:])
else:
    print("Unknown command %s" % (command))

