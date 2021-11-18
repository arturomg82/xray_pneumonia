import torch
import torch.nn as nn
import torchvision.models as models


def get_model(device, model_name, output_dim=2, verbose=True):

    """ Returns the selected pre-trained model. The last layer is replaced by a fully connected layer
       with output of dimension 2. """

    model = None

    if model_name == 'resnet18':

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, output_dim)

    elif model_name == 'resnet50':

        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, output_dim, bias=True)

    elif model_name == 'mobilenet_v2':

        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(1280, output_dim, bias=True)

    elif model_name == 'mobilenet_v3_small':

        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(1024, output_dim, bias=True)

    elif model_name == 'mobilenet_v3_large':

        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(1280, output_dim, bias=True)

    if verbose:
        print('[ Pretrained model loaded ( {} ) ] Output model dimensions = {}'.format(model_name, output_dim))
        print(' ')

    return model.to(device)


def get_model_parameters(model_name, model, last_layer=False, verbose=True):

    """ Get model parameters (either all parameters or only those in the last layer). """

    model_parameters = None

    if last_layer:

        # Set the corresponding flag required_grad=True
        if model_name == 'resnet18':
            model_parameters = model.fc.parameters()
        elif model_name == 'resnet50':
            model_parameters = model.fc.parameters()
        elif model_name == 'mobilenet_v2':
            model_parameters = model.classifier.parameters()
        elif model_name == 'mobilenet_v3_small':
            model_parameters = model.classifier.parameters()
        elif model_name == 'mobilenet_v3_large':
            model_parameters = model.classifier.parameters()

        if verbose:
            print('[ Get model parameters (only last layer) ] Model name = {:s}'.format(model_name))
            print(' ')

    else:

        model_parameters = model.parameters()

        if verbose:
            print('[ Get all model parameters ] Model name = {:s}'.format(model_name))
            print(' ')

    return model_parameters
