#config="configs/statute_classification/statute.kogpt2.yaml"
config="configs/statute_classification/statute.lcube-base.yaml"
export CUDA_VISIBLE_DEVICES=0
python run_model.py $config --mode train
