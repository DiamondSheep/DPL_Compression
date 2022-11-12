import utils.get_cifar100
import utils.get_cifar10
import utils.get_imagenet

def dataloader(dataset, data_path, batch_size, n_workers, distributed=False):
    if dataset == 'cifar10':
        train_loader = utils.get_cifar10.get_training_dataloader(data_path=data_path, 
                                                                batch_size=batch_size, 
                                                                num_workers=n_workers)
        test_loader = utils.get_cifar10.get_test_dataloader(data_path=data_path, 
                                                            batch_size=batch_size, 
                                                            num_workers=n_workers)
        num_classes=10
    elif dataset == 'cifar100':
        train_loader = utils.get_cifar100.get_training_dataloader(data_path=data_path, 
                                                                batch_size=batch_size, 
                                                                num_workers=n_workers)
        test_loader = utils.get_cifar100.get_test_dataloader(data_path=data_path, 
                                                            batch_size=batch_size, 
                                                            num_workers=n_workers)
        num_classes=100
    elif dataset == 'imagenet':
        train_loader = utils.get_imagenet.get_train_dataloader(data_path=data_path, 
                                                            batchsize=batch_size, 
                                                            num_workers=n_workers,
                                                            distributed=distributed)
        test_loader = utils.get_imagenet.get_val_dataloader(data_path=data_path, 
                                                            batchsize=batch_size, 
                                                            num_workers=n_workers)
        num_classes = 1000
    
    return train_loader, test_loader, num_classes