import argparse


Y_DIM = 38
X_DIM = 25


def add_args(parser):
 # model architecture options
    parser.add_argument("--y_dim", type=int, default=Y_DIM)
    parser.add_argument("--x_dim", type=int, default=X_DIM)

    # flow options
    parser.add_argument("--num_steps_maf", type=int, default=20)
    parser.add_argument("--affine_type", type=str, default="softplus", choices=["sigmoid", "softplus", "atan"])
    parser.add_argument("--num_steps_arqs", type=int, default=0)
    parser.add_argument("--num_steps_caf", type=int, default=0)
    parser.add_argument("--coupling_net", type=str, default="mlp", choices=["mlp", "att"])
    parser.add_argument("--att_embed_shape", type=int, default=64)
    parser.add_argument("--att_num_heads", type=int, default=4)
    parser.add_argument("--num_transform_blocks_maf", type=int, default=8)
    parser.add_argument("--num_transform_blocks_arqs", type=int, default=7)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout_probability_maf", type=float, default=0.0)
    parser.add_argument("--dropout_probability_arqs", type=float, default=0.1)
    parser.add_argument("--dropout_probability_caf", type=float, default=0.0)
    parser.add_argument(
        "--use_residual_blocks_maf", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument(
        "--use_residual_blocks_arqs", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument(
        "--batch_norm_maf", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument(
        "--batch_norm_arqs", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument(
        "--batch_norm_caf", type=eval, default=True, choices=[True, False]
    )
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--tail_bound", type=float, default=1.0)
    parser.add_argument("--hidden_dim_maf", type=int, default=128)
    parser.add_argument("--hidden_dim_arqs", type=int, default=300)
    parser.add_argument("--hidden_dim_caf", type=list, default=[128 for _ in range(8)])

    parser.add_argument(
        "--permute_type",
        type=str,
        default="random-permutation",
        choices=["random-permutation", "no-permutation"],
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="alternating-binary",
        choices=["alternating-binary", "block-binary", "identity"],
    )
    parser.add_argument(
        "--init_identity", type=eval, default=True, choices=[True, False]
    )

    # training options
    parser.add_argument("--n_load_cores", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048 * 3,
        help="Batch size (of datasets) for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Latent learning rate for the Adam optimizer.",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Adam.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500, 
        help="Number of epochs for training (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for initializing training. "
    )

    # data options
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=6000000)
    parser.add_argument("--test_start", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=4000000)

    # logging and saving frequency
    parser.add_argument(
        "--log_name", type=str, default="EEMAF_prova", help="Name for the log dir"
    )
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=5)

    # validation options
    parser.add_argument("--validate_at_0", default=False, action="store_true")
    parser.add_argument(
        "--no_validation",
        default=False,
        action="store_true",
        help="Whether to disable validation altogether.",
    )
    parser.add_argument(
        "--save_val_df",
        default=True,
        action="store_true",
        help="Whether to save the validation dataframes.",
    )

    # resuming
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to be loaded.",
    )
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument(
        "--resume_optimizer",
        action="store_true",
        help="Whether to resume the optimizer when resumed training.",
    )

    # device
    parser.add_argument("--device", default="cuda", type=str)
    # distributed training
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed nodes."
    )
    parser.add_argument(
        "--dist_url",
        default="tcp://127.0.0.1:9991",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="GPU id to use. None means using all available GPUs.",
    )

    return parser

def get_parser():
    # command line args
    parser = argparse.ArgumentParser(
        description="Flow-based Point Cloud Generation Experiment"
    )
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
