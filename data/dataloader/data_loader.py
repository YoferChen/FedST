import torchvision.transforms as transforms
from data.dataset import create_dataset
import torch
class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None

def get_dataloader(opt, opt_t, log,train_dataidxs=None, test_dataidxs=None, data_name = [None,None,None,None],if_test = False):
    #opt = unet_options.TrainOptions().parse()
    #opt_t = unet_options.TestOptions().parse()
    #train_dataset
    if opt.cross_validation:
        image_dir = opt.total_img_dir_real
        label_dir=opt.total_label_dir_real
    else:
        image_dir = opt.train_img_dir_real
        label_dir=opt.train_label_dir_real
    #pdb.set_trace()
    train_loader_real = CreateDataLoader(opt, dataroot=opt.dataroot, image_dir=image_dir, \
                                         label_dir=label_dir, \
                                         is_aug=True, train_dataidxs=train_dataidxs,\
                                         image_name = data_name[0],label_name=data_name[1])
    train_dataset_real = train_loader_real.load_data()
    dataset_size_real = len(train_loader_real)
    log.logger.info('#train images = %d' %dataset_size_real)

    #val_dataset
    num_threads = 1
    batch_size = opt.batch_size
    serial_batches = True
    if opt.cross_validation:
        image_dir = opt.total_img_dir_real
        label_dir=opt.total_label_dir_real
    else:
        image_dir = opt.val_img_dir_real
        label_dir=opt.val_label_dir_real
    val_loader = CreateDataLoader(opt, batch_size,serial_batches, num_threads,dataroot=opt.dataroot,\
                                  image_dir=image_dir,label_dir=label_dir,\
                                  record_txt=opt.val_img_list,is_aug=False, \
                                  train_dataidxs=test_dataidxs, image_name = data_name[2],label_name=data_name[3])
    val_dataset = val_loader.load_data()
    val_dataset_size = len(val_loader)
    log.logger.info('#eval images = %d' %val_dataset_size)
    num_threads = 1
    batch_size = 1
    serial_batches = True
    
    if if_test:
        test_loader = CreateDataLoader(opt_t, batch_size,serial_batches, num_threads,dataroot=opt_t.dataroot,\
                                  image_dir=opt_t.test_img_dir,label_dir=opt_t.test_label_dir,\
                                  is_aug=False)

        test_dataset = test_loader.load_data()
        test_dataset_size = len(test_loader)
        log.logger.info('#test images = %d' %test_dataset_size)
    else:
        test_loader = None

    return train_dataset_real, val_dataset, test_loader

def CreateDataLoader(opt,*args,**kwargs):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt,*args,**kwargs)
    return data_loader


# Wrapper class of Dataset class that performs
# multi-threaded data loading
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'
    
    def initialize(self, opt, *args,**kwargs):
        BaseDataLoader.initialize(self, opt)
        if len(args)>0:
            assert len(args)==3,'num error'
            batch_size = min(opt.batch_size,args[0])
            serial_batches = opt.serial_batches or args[1]
            num_threads = min(opt.num_threads,args[2])
        else:
            batch_size = opt.batch_size
            serial_batches = opt.serial_batches
            num_threads = opt.num_threads
        #print('batch_size: ', batch_size)
        self.dataset = create_dataset(opt,**kwargs)
        # print("self.dataset.len() = ", len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=int(num_threads))
        
    def load_data(self):
        return self

    def __len__(self):
        '''
        if self.if_train:
            length = min(len(self.dataset), self.opt.max_dataset_size)
        else:
            length = len(self.dataset)
        '''
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
            
