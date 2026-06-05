conda activate gflower

stein_particles=8
stein_loop=1
stein_step=0.02
stein_kernel=rbf

for flow_matching_type in cfm ot_cfm; do
    for env in halfcheetah hopper walker2d; do
        for dataset in medium-expert medium medium-replay; do
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

            for scale in 0.01 0.1 1.0; do
                python run/eval.py \
                --device cuda:0 \
                --seed 0 \
                --random_repeat 5 \
                --exp_name "$flow_prefix"H20_1e6steps_stein_10steps_inf_K"$stein_particles"_loop"$stein_loop"_step"$stein_step"_scale"$scale" \
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
                --guidance_method stein \
                --grad_compute_at x_1 \
                --grad_wrt x_1 \
                --grad_schedule const \
                --grad_scale $scale \
                --stein_particles $stein_particles \
                --stein_loop $stein_loop \
                --stein_step $stein_step \
                --stein_kernel $stein_kernel
            done
        done
    done
done
