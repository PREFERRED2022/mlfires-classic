cp /data2/ffp/code/mlfires/ML_fires_al/space_newcv_test_nn.py /data2/ffp/code/mlfires/ML_fires_al/space_newcv.py
python fires4_newcrossval.py 2>&1 | tee test_nn_defCV.log
