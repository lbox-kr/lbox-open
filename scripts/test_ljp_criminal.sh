#config="configs/ljp/criminal/ljp.criminal.lcube-base.test.yaml"
#config="configs/ljp/criminal/ljp.criminal.kogpt2.test.yaml"
export CUDA_VISIBLE_DEVICES=0
python run_model.py $config --mode test
