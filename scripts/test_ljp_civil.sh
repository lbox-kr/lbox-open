#config="configs/ljp/civil/ljp.civil.kogpt2.test.yaml"
#config="configs/ljp/civil/ljp.civil.lcube-base.test.yaml"
export CUDA_VISIBLE_DEVICES=0
python run_model.py $config --mode test
