python main.py \
    --gpu 2 --task train \
    --steps 400000  --batch_size 16 --lr 1e-4\
    --expert_h 1 --expert_lockdown 168 --prob_imitation_steps 200000 --base_prob_imitation 0.5 \
    --repeat 24 --rd_no_policy_days 25 --fixed_no_policy_days_list 0 10 20 \
    --mobility_decay 0.99 --L0 72 --H0 3\
    --I_threshold 100 --lockdown_threshold 336 \
    --beta_s 0.1 --beta_m 3 --gamma 0.3 --theta 0.3 \

