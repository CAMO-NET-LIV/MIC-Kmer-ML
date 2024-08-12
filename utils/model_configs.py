from models.cnn import CNN
from models.cnn2 import CNN2
from models.dnp import DNP
from models.kwise import KWise
from models.mlp import MLP


def get_model(args, device):
    model_name = args.model.lower()
    if model_name == 'cnn':
        model = CNN(input_dim=4 ** args.km, device=device)
        input_shape = (1, 4 ** args.km)
    elif model_name == 'cnn2':
        model = CNN2(input_dim=4 ** args.km, device=device)
        input_shape = (1, 4 ** args.km)
    elif model_name == 'mlp':
        model = MLP(input_dim=4 ** args.km)
        input_shape = (4 ** args.km,)
    elif model_name == 'kw':
        model = KWise(input_dim=4 ** args.km, device=device)
        input_shape = (4 ** args.km,)
    elif model_name == 'dnp':
        model = DNP(input_dim=4 ** args.km, output_dim=1)
        input_shape = (4 ** args.km,)
    else:
        raise Exception('Invalid model type')

    return model.to(device), input_shape


def get_model_with_dim(args, device, input_dim):
    model_name = args.model.lower()
    if model_name == 'cnn':
        model = CNN(input_dim=input_dim, device=device)
    elif model_name == 'cnn2':
        model = CNN2(input_dim=input_dim, device=device)
    elif model_name == 'mlp':
        model = MLP(input_dim=input_dim)
    elif model_name == 'kw':
        model = KWise(input_dim=input_dim, device=device)
    elif model_name == 'dnp':
        model = DNP(input_dim=input_dim, output_dim=1)
    else:
        raise Exception('Invalid model type')

    return model.to(device)
