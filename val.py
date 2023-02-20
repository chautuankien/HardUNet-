import argparse
import os
from glob import glob
from collections import OrderedDict
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import torch.nn as nn
from pytorch_model_summary import summary
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
import archs
from dataset import Dataset
from metrics import iou_score
from metrics import dice_coef
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    '''device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model.to(device)'''
    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    #_, val_img_ids = train_test_split(img_ids, test_size=0.1, random_state=41)
    val_img_ids = img_ids
    model.load_state_dict(torch.load('models/TrainDataset_NestedUNet_woDS/model.pth'), strict=True)
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=True)

    avg_meters = {'IoU': AverageMeter(),
                  'Dice': AverageMeter()}

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        with open("models/TrainDataset_NestedUNet_woDS/testDice.csv", "w", newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Test Dice']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)


            for input, target, meta in tqdm(val_loader, total=len(val_loader)):

                input = input.cuda()
                target = target.cuda()

                # compute output
                if config['deep_supervision']:
                    output = model(input)[-1]
                else:
                    output = model(input)
                # print(type(output))
                # print(output.shape)
                iou = iou_score(output, target)
                dice = dice_coef(output, target)
                #writer.writerow({'Test Dice': iou})
                writer.writerow({'Test Dice': dice})
                avg_meters['IoU'].update(iou, input.size(0))
                avg_meters['Dice'].update(dice, input.size(0))

                output = torch.sigmoid(output).cpu().numpy()

                for i in range(len(output)):
                    for c in range(config['num_classes']):
                        cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                    (output[i, c] * 255).astype('uint8'))

    print('Dice: %.4f - IoU %.4f' % (avg_meters['Dice'].avg, avg_meters['IoU'].avg))



    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()