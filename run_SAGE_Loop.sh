# chmod +x run_SAGE_Loop.sh
# ./run_SAGE_Loop.sh

 
# classification dataset credit-g 
python ensemble/classification_ensemble/classification_auto_ensemble.py \
  --llm 'gpt-3.5-turbo' --exam_iterations 2 --feat_iterations 1 --model_iterations 4 --param_iterations 1 --dataset "ds_credit" \
  --enable_optimization --enable_feedback 


## regression dataset concrete 
python ensemble/regression_ensemble/regression_auto_ensemble.py \
  --llm 'gpt-3.5-turbo' --exam_iterations 2 --feat_iterations 1 --model_iterations 4  --dataset "concrete" \
  --enable_feedback
 

## clustering dataset breast
python ensemble/cluster_ensemble/cluster_auto_ensemble.py \
  --llm 'gpt-3.5-turbo' --exam_iterations 2 --feat_iterations 1 --model_iterations 4 --param_iterations 1 --dataset "breast" \
  --enable_optimization --enable_feedback