import argparse
import torch
from models import MSTN_Transformer, MSTN_BiLSTM

def main():
    parser = argparse.ArgumentParser(description='MSTN')

    parser.add_argument('--task_name', type=str, required=True, default='classification')
    parser.add_argument('--model', type=str, required=True, default='MSTN_Transformer')
    parser.add_argument('--data', type=str, required=True, default='PAMAP2')
    parser.add_argument('--root_path', type=str, default='./data/')
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    model_dict = {
        'MSTN_Transformer': MSTN_Transformer,
        'MSTN_BiLSTM': MSTN_BiLSTM,
    }

    model = model_dict[args.model](args).float()
    
    print(f"Model: {args.model} | Task: {args.task_name} | Data: {args.data}")

if __name__ == "__main__":
    main()