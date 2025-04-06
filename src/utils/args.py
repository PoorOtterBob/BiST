import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='')
    parser.add_argument('--model_name', type=str, default='BiST')

    parser.add_argument('--bs', type=int, default=64)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--seq_len', type=int, default=12) # 12, 24, 96
    parser.add_argument('--horizon', type=int, default=12) # 12, 24, 96, 192, 336
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=20)

    parser.add_argument('--core', type=int, default=0) # control the core and core num
    parser.add_argument('--kernel_size', type=int, default=3) # 3, 5, 25

    parser.add_argument('--ct', type=int, default=0) # continue learning

    ## Setting about Experiments
    parser.add_argument('--extra_type', type=int, default=1) # 0 for baseline-only, 1 for joint-training, and 2 for fine-tuning. 
    parser.add_argument('--same', type=int, default=0) # The same ST-Module in ST Model and STRIP or not
    
    ## hyperparameters of Decouple
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--rp_layer', type=int, default=2) # residual propagation layer. 

    ## Residual Propagation Kernel
    parser.add_argument('--use_global_opt', type=bool, default=False)
    parser.add_argument('--mrf', type=int, default=1) # Symmetric Kernel or not
    parser.add_argument('--predefined_adj', type=int, default=0) # Predefined Kernel or not
    parser.add_argument('--datadriven_adj', type=int, default=0) # Adaptive Kernel or not
    parser.add_argument('--datadriven_adj_dim', type=int, default=32)
    parser.add_argument('--datadriven_adj_head', type=int, default=0)
    parser.add_argument('--adaptive_adj', type=int, default=1) # Data-driven Kernel or not
    parser.add_argument('--adaptive_adj_dim', type=int, default=10)

    parser.add_argument('--shap', type=int, default=0)
    parser.add_argument('--layer', type=int, default=3)

    return parser