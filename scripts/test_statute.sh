#config="configs/statute_classification/statute.kogpt2.test.yaml"
#config="configs/statute_classification/statute.lcube-base.test.yaml"
export CUDA_VISIBLE_DEVICES=0
python run_model.py $config --mode test
