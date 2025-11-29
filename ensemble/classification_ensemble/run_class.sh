#!/bin/bash
# run.sh
# chmod +x "$0"   # 如果还没给执行权限可保留

==== 前三条 SAGE-Loop 实验，各 1.5 h =====
timeout 5400 python /home/usr01/cuicui/SAGE-Loop/ensemble/classification_ensemble/code_time_memory_token_parallel_switch.py \
  --llm 'gpt-3.5-turbo' --exam_iterations 3 --feat_iterations 10 --model_iterations 5 --param_iterations 5 --dataset "ds_credit" \
  --enable-feedback

timeout 5400 python /home/usr01/cuicui/SAGE-Loop/ensemble/classification_ensemble/code_time_memory_token_parallel_switch.py \
  --llm 'gpt-3.5-turbo' --exam_iterations 3 --feat_iterations 5 --model_iterations 5 --param_iterations 5 --dataset "cd2" \
  --enable-feedback

timeout 5400 python /home/usr01/cuicui/SAGE-Loop/ensemble/classification_ensemble/code_time_memory_token_parallel_switch.py \
  --llm 'gpt-3.5-turbo' --exam_iterations 5 --feat_iterations 10 --model_iterations 5 --param_iterations 5 --dataset "cf1" \
  --enable-feedback


timeout 5400 python /home/usr01/cuicui/SAGE-Loop/ensemble/classification_ensemble/code_time_memory_token_parallel_switch.py \
  --llm 'gpt-3.5-turbo' --exam_iterations 3 --feat_iterations 10 --model_iterations 5 --param_iterations 5 --dataset "ds_credit" \
  --enable-optimization

timeout 5400 python /home/usr01/cuicui/SAGE-Loop/ensemble/classification_ensemble/code_time_memory_token_parallel_switch.py \
  --llm 'gpt-3.5-turbo' --exam_iterations 3 --feat_iterations 5 --model_iterations 5 --param_iterations 5 --dataset "cd2" \
  --enable-optimization

timeout 5400 python /home/usr01/cuicui/SAGE-Loop/ensemble/classification_ensemble/code_time_memory_token_parallel_switch.py \
  --llm 'gpt-3.5-turbo' --exam_iterations 5 --feat_iterations 10 --model_iterations 5 --param_iterations 5 --dataset "cf1" \
  --enable-optimization

# ==== AutoGluon 实验，4h =====
# timeout 14400 python /home/usr01/cuicui/autoML-ensemble/comparative_experiment/autogluon/autogluon_classification_time_memory.py



# # 如需跑 H2O，同理
# timeout 30000 python /home/usr01/cuicui/autoML-ensemble/comparative_experiment/h2o/h2o_classification_time_memory.py



# # ==== TPOT 实验，4h =====
# timeout 21600 python /home/usr01/cuicui/autoML-ensemble/comparative_experiment/tpot/tpot_classification_time_memory.py