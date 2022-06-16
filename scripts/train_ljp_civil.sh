#config="configs/ljp/civil/ljp.civil.kogpt2.yaml"
config="configs/ljp/civil/ljp.civil.lcube-base.yaml"
export CUDA_VISIBLE_DEVICES=0
python run_model.py $config --mode train
