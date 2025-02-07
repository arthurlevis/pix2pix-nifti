import time
import os
import numpy as np
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options (includes base options)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode (nifti_dataset)
    dataset_size = len(dataset)    # number of slicing windows

    model = create_model(opt)      # create a model given opt.model
    model.setup(opt)
    total_iters = 0

    G_losses = {
                'Lgan': [],  # how well the generator fools the discriminator
                'L1': []     # fake image quality/accuracy
            }

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        # model.update_learning_rate()  # update learning rates in the beginning of every epochs

        for i, data in enumerate(dataset):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset & apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights 

            if total_iters % opt.save_latest_freq == 0:  # model saved every '--save_latest_freq' (default = 5'000)
                # print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix) 

        if epoch % opt.save_epoch_freq == 0:  # model saved every '--save_epoch_freq' (default = 5)
            # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')  # used for testing
            model.save_networks(epoch)     # last epoch = same as latest

            # Losses
            losses = model.get_current_losses()
            # print(f"Epoch {epoch}, G_L1: {losses['G_L1']}")  # generator's L1 loss
            G_losses['Lgan'].append(losses['G_GAN'])
            G_losses['L1'].append(losses['G_L1'])

        model.update_learning_rate()  # update learning rates at the end of every epoch

        print('End of epoch %d / %d \t Time taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    loss_path = os.path.join(opt.checkpoints_dir, opt.name, 'G-losses.npz')
    np.savez(loss_path, **G_losses)

    with open(os.path.join(opt.checkpoints_dir, opt.name, 'total_iters.txt'), 'w') as f:
        f.write(str(total_iters))
