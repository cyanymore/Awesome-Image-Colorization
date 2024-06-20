
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'aligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'MobileAR':
        assert(opt.dataset_mode == 'aligned')
        from .MobileAR import MobileARModel
        model = PerCycleGANModel()
    elif opt.model == 'pan_cycle_gan':
        assert(opt.dataset_mode == 'aligned')
        from .pan_cycle_gan_model import PanCycleGANModel
        model = PanCycleGANModel()
    elif opt.model == 'attention_gan':
        assert(opt.dataset_mode == 'aligned')
        from .attention_gan_model import AttentionGANModel
        model = AttentionGANModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
