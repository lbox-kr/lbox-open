config="configs/summarization/summarization.legal-mt5s.test.yaml"
export CUDA_VISIBLE_DEVICES=0
python run_model.py $config --mode test
