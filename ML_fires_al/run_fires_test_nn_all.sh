export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['BA']" do True
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['hybrid1']" do
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['f1-score 1', 'auc']" do
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py 
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['hybrid2', 'hybrid5']" do True
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['NH2', 'NH5']" do True 
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_nn.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_test.py "['NH10']" do


