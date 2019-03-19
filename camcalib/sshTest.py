import subprocess as sp
import sys


pipe0 = sp.Popen(['ssh', 'pi@10.0.0.3', '-p', '6622', '~/stream.sh'], stdout = sp.PIPE )
