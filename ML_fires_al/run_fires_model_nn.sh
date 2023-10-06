export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
cp /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test_model.py /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/space_new_test.py
python -u /mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/fires_hyperopt_newdata.py


