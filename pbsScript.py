import os, sys

#parameters being tuned
p_arr = [0.5, 1.0, 1.5, 2.0]
q_arr = [0.5, 1.0, 1.5, 2.0]
restart_arr = [0.001, 0.0005, 0.005]

#MAIN.PY
for p in p_arr:
    for q in q_arr:
        filename = 'lightProject_' + 'p_' + str(p) + 'q_' + str(q)
        with open(filename, 'w') as fp:
            fp.writelines(['#PBS -N lightGroup\n',      
                '#PBS -A rthind3\n',             
                '#PBS -l nodes=1:ppn=10:gpus=1\n',     
                '#PBS -l pmem=5gb\n',            
                '#PBS -l walltime=02:00:00\n',  
                '#PBS -q coc-ice-gpu\n',
                '#PBS -j oe\n',                 
                '#PBS -o ' + filename + 'Out.out\n\n',

                'cd $PBS_O_WORKDIR\n',

                'module load pytorch/1.11.0\n',

                'python3 main.py --data gowalla --lambda2 0' + '--p ' + str(p) + ' --q_val ' + str(q)
                ])
        print('Generaged: ', filename)
        os.system('qsub '+filename)
        print('Job submitted')

#MAIN2.PY
for p in p_arr:
    for q in q_arr:
        for restart in restart_arr:
            filename = 'lightProject_' + 'p_' + str(p) + 'q_' + str(q)
            with open(filename, 'w') as fp:
                fp.writelines(['#PBS -N lightGroup\n',      
                    '#PBS -A rthind3\n',             
                    '#PBS -l nodes=1:ppn=10:gpus=1\n',     
                    '#PBS -l pmem=5gb\n',            
                    '#PBS -l walltime=02:00:00\n',  
                    '#PBS -q coc-ice-gpu\n',
                    '#PBS -j oe\n',                 
                    '#PBS -o ' + filename + 'Out.out\n\n',

                    'cd $PBS_O_WORKDIR\n',

                    'module load pytorch/1.11.0\n',

                    'python3 main2.py --data gowalla --lambda2 0' + '--p ' + str(p) + ' --q_val ' + str(q) + ' --restart ' + str(restart)
                    ])
            print('Generaged: ', filename)
            os.system('qsub '+filename)
            print('Job submitted')