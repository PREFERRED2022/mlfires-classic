export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
#cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/$1 /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/cl_space_new_test.py
#python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test_cl.py "['BA']" $2 False $3
#cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/$1 /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/cl_space_new_test.py
#python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test_cl.py "['hybrid1']" $2 False $3
#cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/$1 /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/cl_space_new_test.py
#python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test_cl.py "['f1-score 1', 'auc']" $2 False $3
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/$1 /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py 
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test_cl.py "['hybrid2', 'hybrid5']" $2 False $3
#cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/$1 /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/cl_space_new_test.py
#python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test_cl.py "['NH2', 'NH5']" $2 False $3
#cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/$1 /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/cl_space_new_test.py
#python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test_cl.py "['NH10']" $2 False $3
