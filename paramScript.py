import os, sys
from parseMain import start
from parseMain2 import start2
from parser import args

#parameters being tuned
p_arr = [0.5, 1.0, 1.5, 2.0]
q_arr = [0.5, 1.0, 1.5, 2.0]
restart_arr = [0.001, 0.0005, 0.005]

#MAIN.PY
for p in p_arr:
    for q in q_arr:
        start(args, p, q)

#MAIN2.PY
for restart in restart_arr:
    start2(args, restart)