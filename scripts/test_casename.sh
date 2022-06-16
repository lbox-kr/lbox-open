#config="configs/casename_classification/casename.kogpt2.test.yaml"
#config="configs/casename_classification/casename.lcube-base.test.yaml"
export CUDA_VISIBLE_DEVICES=0
python run_model.py $config --mode test

