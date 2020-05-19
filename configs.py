import copy

config_cifar10 = {}
config_cifar10['n_classes'] = 10
config_cifar10['dataset'] = 'cifar10'
config_cifar10['epoch_num'] = 50
config_cifar10['zdim'] = 64
config_cifar10['lr'] = 5e-4
config_cifar10['e_arch'] = 'dcgan'
config_cifar10['g_arch'] = 'dcgan_mod'
config_cifar10['verbose'] = True
config_cifar10['save_every_epoch'] = 10

config_cifar10['optimizer'] = 'adam'
config_cifar10['adam_beta1'] = 0.5
config_cifar10['lr_schedule'] = 'manual_smooth'
config_cifar10['batch_size'] = 50
config_cifar10['init_std'] = 0.0099999
config_cifar10['init_bias'] = 0.0
config_cifar10['batch_norm'] = True
config_cifar10['batch_norm_eps'] = 1e-05
config_cifar10['batch_norm_decay'] = 0.9
config_cifar10['conv_filters_dim'] = 5
config_cifar10['e_pretrain'] = True
config_cifar10['e_pretrain_sample_size'] = 256
config_cifar10['e_noise'] = 'add_noise'
config_cifar10['e_num_filters'] = 1024
config_cifar10['e_num_layers'] = 4
config_cifar10['g_num_filters'] = 1024
config_cifar10['g_num_layers'] = 4
config_cifar10['d_num_layers'] = 4
config_cifar10['d_num_filters'] = 1024

config_cifar10['cost'] = 'l2sq'
config_cifar10['lambda'] = 1.

config_cifar10['dratio_mode'] = 'rebalance'
config_cifar10['gratio_mode'] = 'rebalance'
config_cifar10['wgan_d_penalty'] = True

config_SVHN = copy.deepcopy(config_cifar10)
config_SVHN['dataset'] = 'svhn'
config_SVHN['n_classes'] = 10
config_SVHN['epoch_num'] = 80

config_mnist = {}
config_mnist['dataset'] = 'mnist'
config_mnist['verbose'] = True
config_mnist['save_every_epoch'] = 10

config_mnist['optimizer'] = 'adam'
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 5e-4
config_mnist['lr_schedule'] = 'none'
config_mnist['batch_size'] = 100
config_mnist['epoch_num'] = 50
config_mnist['init_std'] = 0.0099999
config_mnist['init_bias'] = 0.0
config_mnist['batch_norm'] = True
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_decay'] = 0.9
config_mnist['conv_filters_dim'] = 4

config_mnist['e_pretrain'] = True
config_mnist['e_pretrain_sample_size'] = 1000
config_mnist['e_noise'] = 'add_noise'
config_mnist['e_num_filters'] = 1024
config_mnist['e_num_layers'] = 4
config_mnist['e_arch'] = 'dcgan'
config_mnist['g_num_filters'] = 1024
config_mnist['g_num_layers'] = 3
config_mnist['g_arch'] = 'dcgan_mod'
config_mnist['d_num_filters'] = 512
config_mnist['d_num_layers'] = 4

config_mnist['zdim'] = 64
config_mnist['cost'] = 'l2sq'
config_mnist['lambda'] = 10.
config_mnist['n_classes'] = 10

config_mnist['dratio_mode'] = 'rebalance'
config_mnist['gratio_mode'] = 'rebalance'

config_mnist['wgan_d_penalty'] = True

config_fashion = copy.deepcopy(config_mnist)
config_fashion['zdim'] = 64
config_fashion['dataset'] = 'fashion'
