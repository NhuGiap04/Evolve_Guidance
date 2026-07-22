set -e

export MPLBACKEND=${MPLBACKEND:-Agg}

if [ "${CONDA_DEFAULT_ENV:-}" != "gflower" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate gflower
fi

# Restrict the particle sweep with, for example, SMC_PARTICLES_VALUES="8 32".
smc_particles_values=${SMC_PARTICLES_VALUES:-"8 32"}
smc_scale=${SMC_SCALE:-0.01}
smc_ess_threshold=${SMC_ESS_THRESHOLD:-0.5}
smc_resample_every=${SMC_RESAMPLE_EVERY:-1}

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

            for num_particles in $smc_particles_values; do
                python run/eval.py \
                    --device cuda:0 \
                    --seed 0 \
                    --random_repeat 5 \
                    --exp_name "$flow_prefix"H20_1e6steps_smc_10steps_inf_K"$num_particles"_scale"$smc_scale"_ess"$smc_ess_threshold" \
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
                    --guidance_method smc \
                    --batch_size 1 \
                    --smc_particles "$num_particles" \
                    --smc_scale "$smc_scale" \
                    --smc_ess_threshold "$smc_ess_threshold" \
                    --smc_resample_every "$smc_resample_every" \
                    "$@"
            done
        done
    done
done
