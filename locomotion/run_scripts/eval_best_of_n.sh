set -e

export MPLBACKEND=${MPLBACKEND:-Agg}

if [ "${CONDA_DEFAULT_ENV:-}" != "gflower" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate gflower
fi

best_of_n_values=${BEST_OF_N_VALUES:-"8 32"}

# Additional eval options may be appended on the command line, for example:
# bash run_scripts/eval_best_of_n.sh --random_repeat 1 --max_episode_length 10
for flow_matching_type in cfm ot_cfm; do
    for env in halfcheetah hopper walker2d; do
        for dataset in medium-expert medium medium-replay; do
            if [ "$env" = "halfcheetah" ]; then
                state_dim=17
                action_dim=6
            elif [ "$env" = "hopper" ]; then
                state_dim=11
                action_dim=3
            elif [ "$env" = "walker2d" ]; then
                state_dim=17
                action_dim=6
            fi

            if [ "$flow_matching_type" = "cfm" ]; then
                flow_prefix=""
            elif [ "$flow_matching_type" = "ot_cfm" ]; then
                flow_prefix="ot_"
            fi

            for num_candidates in $best_of_n_values; do
                python run/eval.py \
                    --device cuda:0 \
                    --seed 0 \
                    --random_repeat 5 \
                    --exp_name "$flow_prefix"H20_1e6steps_best_of_"$num_candidates"_10steps_inf \
                    --env "$env-$dataset-v2" \
                    --state_dim "$state_dim" \
                    --action_dim "$action_dim" \
                    --horizon 20 \
                    --flow_exp_name "$flow_prefix"H20_1e6steps \
                    --flow_cp 19 \
                    --flow_matching_type "$flow_matching_type" \
                    --value_exp_name H20_inf \
                    --value_cp 2 \
                    --ode_t_steps 10 \
                    --guidance_method best_of_n \
                    --batch_size 1 \
                    --best_of_n "$num_candidates" \
                    "$@"
            done
        done
    done
done
