import os
import argparse
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.bist import BiST, MLP
from src.base.engine import BaseEngine
from src.utils.args import get_config
from src.utils.dataloader import load_dataset, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def cont_learning(model, save_path, args):
    filename = 'final_model_s{}_c{}_same{}.pt'.format(args.seed, 
                                                      args.core, 
                                                      args.same)
    model.load_state_dict(torch.load(
        os.path.join(save_path, filename), map_location=args.device))
    return model


def get_config():
    parser = get_config()
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--prompt_dim', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    
    # tranining parameters
    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record.log')
    logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    seed = torch.randint(999999, (1,)) # set random seed here
    set_seed(seed)
    device = torch.device(args.device)
    
    data_path, tod_size, node_num = get_dataset_info(args.dataset)
    args.node_num = node_num
    dataloader, scaler = load_dataset(data_path, args, logger)

    base = MLP(node_num=node_num,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                num_layer=args.num_layer, 
                model_dim=args.model_dim, 
                prompt_dim=args.prompt_dim, 
                tod_size=tod_size, 
                kernel_size=args.kernel_size)
    model_dim = args.model_dim + 3*args.prompt_dim
    model = BiST(node_num=node_num,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 model_args=vars(args),
                 stmodel=base,
                 dim=[model_dim, model_dim],
                 core=args.core,
                 )
    if args.ct:
        try:
            model = cont_learning(model, log_dir, args)
        except:
            print('No pretrained model!')
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[30, 60, 80, 100], 
                                                     gamma=0.5)

    engine = BaseEngine(device=device,
                        model=model,
                        dataloader=dataloader,
                        scaler=scaler,
                        sampler=None,
                        loss_fn=loss_fn,
                        lrate=args.lrate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        clip_grad_value=args.clip_grad_value,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        args=args,)


    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
