def create_model(args, dataset):
    if args.model == "cycle":
        from models.architecture import GAN_model
        return GAN_model(args, dataset)
    elif args.model == "deep":
        from models.architecture2 import DEEP_model
        return DEEP_model(args, dataset)
