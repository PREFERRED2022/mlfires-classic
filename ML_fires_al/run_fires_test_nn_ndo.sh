export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['BA']" ndo True
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['hybrid1']" ndo
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['f1-score 1', 'auc']" ndo
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py 
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['hybrid2', 'hybrid5']" ndo True
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['NH2', 'NH5']" ndo True 
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['NH10']" ndo


