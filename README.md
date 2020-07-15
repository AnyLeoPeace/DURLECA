# DURLECA
Here is the released code for the DUal-objective Reinforcement-Learning Epidemic Control Agent (DURLECA) presented in our KDD paper *Reinforced Epidemic Control: Saving Both Lives and Economy*.

## TODO
- [ ] Provide a sample code for generating a synthetic OD dataset.


## Generate a synthetic OD dataset
Please note that we would not provide the real-world Beijing dataset due to privacy and ethical concerns. Thus, we suggest users generate a synthetic OD dataset or use any other dataset from their own resources.

Please change the dataset path in the utils.py file.

## Train
```
  python main.py \
    --gpu 2 --task train \
    --steps 400000  --batch_size 16 --lr 1e-4\
    --expert_h 1 --expert_lockdown 168 --prob_imitation_steps 200000 --base_prob_imitation 0.5 \
    --repeat 24 --rd_no_policy_days 25 --fixed_no_policy_days_list 0 10 20 \
    --mobility_decay 0.99 --km 72 --H0 3\
    --I_threshold 100 --lockdown_threshold 336 \
    --beta_s 0.1 --beta_m 3 --gamma 0.3 --theta 0.3 \

```
Users could also adjust parameters on their own to simulate different diseases or to have different objectives.

## Test
```
  python main.py \
    --gpu 0 --task test_list --verbose True --save_path [YOUR PATH] \
    --fixed_no_policy_days_list 20\
```
