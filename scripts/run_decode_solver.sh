CUDA_VISIBLE_DEVICES=0 python -u run_decode_solver.py \
--model_dir diffusion_models/diffuseq_bugfix_h64_lr0.0001_t4000_sqrt_lossaware_seed102_bugfix20240214-00:17:34 \
--seed 102 \
--bsz 100 \
--step 100 \
--split test
