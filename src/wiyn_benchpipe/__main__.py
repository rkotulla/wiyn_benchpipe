import os
import sys

from .executables import *

args = sys.argv[1:]
try:
    command = args[0]
except:
    print("Unknown error")
    sys.exit(-1)


exec = None
if (command == "reduce"):
    exec = wiyn_benchpipe
elif (command == "region"):
    exec = region_file_from_output
elif (command == "sparsepaksim"):
    exec = sparsepak_simulator
elif (command == "preview"):
    exec = grating_preview
elif (command == "model2d"):
    exec = grating_model_2d
else:
    print("Unknown command %s" % (command))
    sys.exit(-1)

# run command
return_value = exec(args[1:])
sys.exit(return_value)

