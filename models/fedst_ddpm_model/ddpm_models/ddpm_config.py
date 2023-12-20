from core.logger import InfoLogger
import core.praser as Praser
from models.fedst_ddpm_model.ddpm_models.models import define_network, define_loss

# ddpm_opt = Praser.parse("models/fedst_ddpm_model/ddpm_models/config/labeltoimage.json")
ddpm_opt = Praser.parse("models/fedst_ddpm_model/ddpm_models/config/labeltoimage_aaf_face_new.json")

def get_netG_and_losses(opt):
    ''' set logger '''
    opt['phase'] = 'train'
    phase_logger = InfoLogger(opt)
    netG = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
    ''' set metrics, loss, optimizer and  schedulers '''
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]
    return netG[0], losses[0]