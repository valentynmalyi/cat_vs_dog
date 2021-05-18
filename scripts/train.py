from cat_vs_dog.train import train_and_save_model
from cat_vs_dog.train.parser import parser

if __name__ == '__main__':
    args = parser.parse_args()
    train_and_save_model(path_model=args.path_model)
