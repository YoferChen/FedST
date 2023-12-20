import importlib
import torch
import collections


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    if 'fedst_ddpm' in model_name:
        model_filename = "models.fedst_ddpm_model." + model_name + "_model"
    elif 'unet' in model_name:
        model_filename = "models.unet_model." + model_name + "_model"
    else:
        model_filename = "models.unet_model." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():

        if name.lower() == target_model_name.lower() and issubclass(cls, torch.nn.Module):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    from torch import nn
    import torch
    model = find_model_using_name(opt.model.lower())
    instance = model()
    instance.initialize(opt)

    if opt.model == 'unet' and opt.mode == 'Train' and opt.pretrained_model is not None:
        # pretrain_model_path = './dataset/Face_segment/model_fedavg/model43_folds0_best.pkl'
        pretrain_model_path = opt.pretrained_model
        print('[For Train] Pretrain model {} has loaded. '.format(pretrain_model_path))

        state_dict = torch.load(pretrain_model_path)
        if 'module' in state_dict:
            state_dict = state_dict['module']

        sd = collections.OrderedDict()

        for i in state_dict.keys():
            if 'net' in i:
                key = '.'.join(i.split('.')[1:])
                sd[key] = state_dict[i]
            else:
                sd[i] = state_dict[i]
        instance.net.load_state_dict(sd, strict=True)
    print("model [%s] was created" % (instance.name()))
    return instance
