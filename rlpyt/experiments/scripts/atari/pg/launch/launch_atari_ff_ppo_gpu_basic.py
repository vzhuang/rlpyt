
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/atari/pg/train/atari_ff_ppo_gpu.py"
affinity_code = encode_affinity(
    n_cpu_core=3,
    n_gpu=3,
    hyperthread_offset=8,
    n_socket=1,
    # cpu_per_run=2,
)
runs_per_setting = 2
experiment_title = "pong_ppo_gridsearch"
variant_levels = list()

noise_classes = ["levy"]
values = list(zip(noise_classes))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "noise_class")]
variant_levels.append(VariantLevel(keys, values, dir_names))

noise_scales = [0.1, 0.5, 1., 5.]
values = list(zip(noise_scales))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "noise_scale")]
variant_levels.append(VariantLevel(keys, values, dir_names))

noise_nonzero_onlys = [True, False]
values = list(zip(noise_nonzero_onlys))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "noise_nonzero_only")]
variant_levels.append(VariantLevel(keys, values, dir_names))

games = ["pong"]
values = list(zip(games))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "game")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

default_config_key = "0"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
