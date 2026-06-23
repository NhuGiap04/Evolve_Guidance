set -e

export MPLBACKEND=${MPLBACKEND:-Agg}

if [ "${CONDA_DEFAULT_ENV:-}" != "gflower" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate gflower
fi

grad_particles=8

for flow_matching_type in cfm ot_cfm; do                                    # cfm otcfm
    for env in halfcheetah hopper walker2d; do                              # halfcheetah hopper walker2d
        for dataset in medium-expert medium medium-replay; do               # medium-expert medium medium-replay
            for grad_at in x_1 x_t; do                                      # x1 xt
                for grad_to in x_1 x_t; do                                  # x1 xt

                 if [ $grad_at == "x_t" ] && [ $grad_to == "x_1" ]; then
                        echo "Skipping x_t -> x_1"
                        continue
                    fi

                    if [ $env == "halfcheetah" ]; then
                        state_dim=17
                        action_dim=6
                    elif [ $env == "hopper" ]; then
                        state_dim=11
                        action_dim=3
                    elif [ $env == "walker2d" ]; then
                        state_dim=17
                        action_dim=6
                    fi

                    if [ $flow_matching_type == "cfm" ]; then
                        flow_prefix=""
                    elif [ $flow_matching_type == "ot_cfm" ]; then
                        flow_prefix="ot_"
                    fi

                    for schedule in cosine_decay const linear_decay exp_decay; do # cosine_delay const linear_decay exp_decay
                        for scale in 0.0 0.01 0.1 1.0; do        # 0.0 0.01 0.1 1.0
                            python run/eval.py \
                            --device cuda:0 \
                            --seed 0 \
                            --random_repeat 5 \
                            --exp_name "$flow_prefix"H20_1e6steps_gradient_10steps_inf_K"$grad_particles"_"$scale"_"$schedule"_grad_at_"$grad_at"_grad_to_"$grad_to" \
                            --env $env-$dataset-v2 \
                            --state_dim $state_dim \
                            --action_dim $action_dim \
                            --horizon 20 \
                            --flow_exp_name "$flow_prefix"H20_1e6steps \
                            --flow_cp 19 \
                            --flow_matching_type $flow_matching_type \
                            --value_exp_name H20_inf \
                            --value_cp 2 \
                            --ode_t_steps 10 \
                            --guidance_method gradient \
                            --batch_size $grad_particles \
                            --grad_compute_at $grad_at \
                            --grad_wrt $grad_to \
                            --grad_schedule $schedule \
                            --grad_scale $scale \
                            "$@"
                        done
                    done
                done
            done
        done
    done
done
