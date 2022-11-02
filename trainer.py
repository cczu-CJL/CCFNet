import argparse
# import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DiceLoss
from utils import gaiDiceLoss

from torchvision import transforms

import logging
from utils import test_single_volume
def inference(best_performance, best_mean_hd95, epoch_num, args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=24, shuffle=False, num_workers=8, pin_memory=False)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('epoch : %d idx %d case %s mean_dice %f mean_hd95 %f' % (epoch_num, i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance : mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    if performance > best_performance:
        best_performance = performance
        best_mean_hd95 = mean_hd95
        change = True
    else:
        change = False

    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (best_performance, best_mean_hd95))
    return performance, mean_hd95, best_performance, best_mean_hd95,change

def trainer_synapse(args, model, snapshot_path, times, strat=0, gai=False):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = CrossEntropyLoss()
    if gai:
        dice_loss = gaiDiceLoss(num_classes)
    else:
        dice_loss = DiceLoss(num_classes)


    if args.is_reload_path:
        model.load_state_dict(torch.load(snapshot_path+args.reload_path, map_location=torch.device('cpu')))


    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    snapshot_name = snapshot_path.split('/')[-1]

    iter_num = strat * len(trainloader)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    best_mean_hd95 = 0.0

    dice_loss_list=[]
    hd_loss_list=[]

    iterator = tqdm(range(strat, max_epoch), ncols=70)   # 进度条
    for epoch_num in iterator:
        model.train()

        a=b=c=0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # 获取数据
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()  # 送入gpu

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)         # 记录数据
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)


            logging.info('iteration %d : loss : %f, loss_dice : %f, loss_ce: %f' % (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))  # 打印数据

            a=a+loss.item()
            b=b+loss_dice.item()
            c=c+loss_ce.item()

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)

                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        a=a/len(trainloader)
        b=b/len(trainloader)
        c=c/len(trainloader)
        logging.info('epoch: %d / %d , loss : %f, loss_dice : %f, loss_ce: %f' % (epoch_num, max_epoch, a, b, c))  # 打印数据

        save_mode_path = os.path.join(snapshot_path, 'interim.pth')
        torch.save(model.state_dict(), save_mode_path)
        # save model

        if epoch_num > 500 and (epoch_num + 1) % 25 == 0:

            # test
            snapshot = os.path.join(snapshot_path, 'interim.pth')
            model.load_state_dict(torch.load(snapshot))
            if args.is_savenii:
                args.test_save_dir = '../predictions'
                test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
                os.makedirs(test_save_path, exist_ok=True)
            else:
                test_save_path = None
            dice_loss_iter, hd_loss_iter, best_performance, best_mean_hd95, change = inference(best_performance, best_mean_hd95, epoch_num,
                                                                         args, model, test_save_path)

            if change:
                save_mode_path = os.path.join(snapshot_path, 'best.pth')
                torch.save(model.state_dict(), save_mode_path)

            dice_loss_list.append(dice_loss_iter)
            hd_loss_list.append(hd_loss_iter)


        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)

            times_save_mode_path = os.path.join(snapshot_path, str(times) + 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), times_save_mode_path)

            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    for i in range(len(dice_loss_list)):
        print(i, dice_loss_list[i], hd_loss_list[i])
    writer.close()
    print("Training Finished!")

    return "Training Finished!"
