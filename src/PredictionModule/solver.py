import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from CMUNet.model import CMUNet
from util.seed import seed_torch
from collections import OrderedDict
from torch.optim import lr_scheduler
from CMUNet.metrics import iou_score
from CMUNet.metrics import AverageMeter
from sklearn.model_selection import LeaveOneOut
from CMUNet.dataset import GeneDataset, TumorDataset
from CMUNet.metrics import calculate_correlations, SavePredictMask



class TumorSolver:
    def __init__(
        self,
        seed = 42,
        mask_ch=1, img_ch=768, epochs=200, lr=0.001, weight_decay=1e-4, verbose=False
    ):
        self.seed = seed
        self.mask_ch = mask_ch
        self.img_ch = img_ch
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        seed_torch(seed)
        
    def _train(self, train_loader, model, criterion, optimizer):
        avg_meters = {'loss': AverageMeter(),
                    'iou': AverageMeter()}
        # train mode
        model.train()
        for input, target, _ in train_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _, _, _, _, _ = iou_score(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg)
                            ])


    def _validate(self, val_loader, model, criterion):
        avg_meters = {'loss': AverageMeter(),
                    'iou': AverageMeter(),
                    'dice': AverageMeter(),
                    'SE':AverageMeter(),
                    'PC':AverageMeter(),
                    'F1':AverageMeter(),
                    'SP':AverageMeter(),
                    'ACC':AverageMeter()
                    }
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for input, target, _ in val_loader:
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                iou, dice, SE, PC, F1, SP, ACC = iou_score(output, target)
                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))
                avg_meters['dice'].update(dice, input.size(0))
                avg_meters['SE'].update(SE, input.size(0))
                avg_meters['PC'].update(PC, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['SP'].update(SP, input.size(0))
                avg_meters['ACC'].update(ACC, input.size(0))

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('SE', avg_meters['SE'].avg),
                            ('PC', avg_meters['PC'].avg),
                            ('F1', avg_meters['F1'].avg),
                            ('SP', avg_meters['SP'].avg),
                            ('ACC', avg_meters['ACC'].avg)
                            ])
        
        
    def _fit_loo(
        self,
        train_loader, val_loader, val_id, out_dir, use_best
    ):
        log = OrderedDict([
                ('epoch', []),
                ('lr', []),
                ('loss', []),
                ('iou', []),
                ('val_loss', []),
                ('val_iou', []),
                ('val_dice', []),
                ('val_SE', []),
                ('val_PC', []),
                ('val_F1', []),
                ('val_SP', []),
                ('val_ACC', [])
            ])

        best_model_dir = os.path.join(out_dir, 'best_model')
        last_model_dir = os.path.join(out_dir, 'last_model')
        log_file_dir = os.path.join(out_dir, 'log')
        
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir,exist_ok=True)
        if not os.path.exists(last_model_dir):
            os.makedirs(last_model_dir,exist_ok=True)
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir,exist_ok=True)
        
        cmu_model = CMUNet(img_ch=self.img_ch, output_ch=self.mask_ch, l=7, k=7)
        cmu_model = cmu_model.cuda()
        params = filter(lambda p: p.requires_grad, cmu_model.parameters())

        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        criterion = nn.BCEWithLogitsLoss().cuda()
        time_start = time.time()
        for epoch in range(self.epochs): 
            # Train for one epoch
            train_log = self._train(train_loader, cmu_model, criterion, optimizer)
            # Evaluate on validation set
            val_log = self._validate(val_loader, cmu_model, criterion)
            scheduler.step()

            log['epoch'].append(epoch + 1)
            log['lr'].append(scheduler.get_last_lr()[0])
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])
            log['val_SE'].append(val_log['SE'])
            log['val_PC'].append(val_log['PC'])
            log['val_F1'].append(val_log['F1'])
            log['val_SP'].append(val_log['SP'])
            log['val_ACC'].append(val_log['ACC'])

            log_file_path = os.path.join(log_file_dir, val_id+'_log.csv')
            pd.DataFrame(log).to_csv(log_file_path, index=False)

            # Save the model if it achieves the best validation IOU
            best_model_path = os.path.join(best_model_dir, val_id+'_best_model.pth')
            if epoch == 20:
                torch.save(cmu_model.state_dict(), best_model_path)
                best_valiou = val_log['iou']
            elif epoch > 20 and val_log['iou'] > best_valiou:
                torch.save(cmu_model.state_dict(), best_model_path)
                best_valiou = val_log['iou']
                if self.verbose:
                    print("=> saved best model")

            torch.cuda.empty_cache()

            time_end = time.time()
            time_cost = time_end - time_start
            if self.verbose:
                print('Epoch [%d/%d]' % (epoch + 1, self.epochs))
                print('loss %.4f - val_loss %.4f - iou %.4f - val_iou %.4f - dice %.4f - SE %.4f - PC %.4f - F1 %.4f - SP %.4f - ACC %.4f'
                    % (train_log['loss'], val_log['loss'], train_log['iou'], val_log['iou'],
                    val_log['dice'], val_log['SE'], val_log['PC'], val_log['F1'], val_log['SP'], val_log['ACC']))
                print('Epoch%d Total Time Cost: %f min' % ((epoch + 1), (time_cost / 60)))
        
        #save last model
        last_model_path = os.path.join(last_model_dir, val_id+'_%d_model.pth' % self.epochs)
        torch.save(cmu_model.state_dict(), last_model_path)
        
        if use_best:
            cmu_model.load_state_dict(torch.load(best_model_path))
        return cmu_model


    def _fit_all(
        self,
        train_loader, out_dir
    ):
        log = OrderedDict([
                ('epoch', []),
                ('lr', []),
                ('loss', []),
                ('iou', [])
            ])
        
        cmu_model = CMUNet(img_ch=self.img_ch, output_ch=self.mask_ch, l=7, k=7)
        cmu_model = cmu_model.cuda()
        params = filter(lambda p: p.requires_grad, cmu_model.parameters())

        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        criterion = nn.BCEWithLogitsLoss().cuda()
        time_start = time.time()
        with tqdm(
            total=self.epochs,
            desc="Training",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        ) as pbar:
            for epoch in range(self.epochs): 
                pbar.set_description(desc=f"Training: Epoch {epoch+1}")
                # Train for one epoch
                train_log = self._train(train_loader, cmu_model, criterion, optimizer)
                scheduler.step()

                log['epoch'].append(epoch + 1)
                log['lr'].append(scheduler.get_last_lr()[0])
                log['loss'].append(train_log['loss'])
                log['iou'].append(train_log['iou'])

                log_file_path = os.path.join(out_dir, 'train_log.csv')
                pd.DataFrame(log).to_csv(log_file_path, index=False)

                torch.cuda.empty_cache()
                pbar.update(1)
                time_end = time.time()
                time_cost = time_end - time_start
                if self.verbose:
                    print('Epoch [%d/%d]' % (epoch + 1, self.epochs))
                    print('loss %.4f - val_loss %.4f - iou %.4f' % (train_log['loss'], train_log['iou']))
                    print('Epoch%d Total Time Cost: %f min' % ((epoch + 1), (time_cost / 60)))
        
        #save last model
        last_model_path = os.path.join(out_dir, '%d_model.pth' % self.epochs)
        torch.save(cmu_model.state_dict(), last_model_path)
        
        return cmu_model



    def train_loo(
        self,
        sample_list : list,
        all_sample_features : list,
        batch_size : int = 5,
        mask_dir : str = './data/mask_png',
        mask_ext : str = '.png',
        HE_path : str = './data/HE',
        HE_ext : str = '.jpg',
        out_dir='./checkpoint_tumor_loo/',
        use_best=False
    ):
        seed_torch(self.seed)
        current_datetime = datetime.now().strftime('%Y%m%d%H%M')
        out_dir = os.path.join(out_dir, current_datetime)
        print('Log and model weights will be saved at',out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir,exist_ok=True)
        train_loo_args = {
            'batch_size': batch_size,
            'mask_dir': mask_dir,
            'mask_ext': mask_ext,
            'HE_path': HE_path,
            'HE_ext': HE_ext,
            'out_dir': out_dir
        }
        param_file = os.path.join(out_dir, 'train_loo_args.txt')
        with open(param_file, 'w') as file:
            file.write(f"Method called at: {current_datetime}\n")
            for arg, value in self.__dict__.items():
                file.write(f"{arg}: {value}\n")
            file.write("\n")
            for arg, value in train_loo_args.items():
                file.write(f"{arg}: {value}\n")
        predict_dir = os.path.join(out_dir, 'predict_masks')
        if not os.path.exists(predict_dir):
            os.makedirs(predict_dir,exist_ok=True)
            
        all_predict_masks = []
        predict_masks = []
        loo = LeaveOneOut()
        with tqdm(
                    total=len(sample_list),
                    desc="Training",
                    bar_format="{l_bar}{bar} [ time left: {remaining} ]",
                ) as pbar:
            for train_idx, val_idx in loo.split(sample_list):
                val_id = sample_list[val_idx[0]]
                train_dataset = TumorDataset(
                    img_ids=[sample_list[i] for i in train_idx],
                    tensor_lists=[all_sample_features[i] for i in train_idx],
                    mask_dir=mask_dir,
                    mask_ext=mask_ext,
                    num_classes=self.mask_ch)
                val_dataset = TumorDataset(
                    img_ids=[sample_list[i] for i in val_idx],
                    tensor_lists=[all_sample_features[i] for i in val_idx],
                    mask_dir=mask_dir,
                    mask_ext=mask_ext,
                    num_classes=self.mask_ch)

                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=False)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False)

                cmu_model = self._fit_loo(train_loader, val_loader, val_id, out_dir, use_best)

                all_mask_predict, mask_predict = SavePredictMask(val_loader,val_id,cmu_model,
                                                                HE_path = HE_path, HE_ext = HE_ext,
                                                                mask_path = mask_dir, mask_ext = mask_ext,
                                                                out_dir=predict_dir)
                all_predict_masks.append(all_mask_predict)
                predict_masks.append(mask_predict)
                
                pbar.update(1)
                
        # mask_predict has 0.5 cutoff; all_predict_masks does not have cutoff
        all_predict_mask_file = os.path.join(predict_dir,'all_predict_masks.pkl')
        predict_mask_file = os.path.join(predict_dir,'predict_masks.pkl')
        
        with open(all_predict_mask_file, 'wb') as f:
            pickle.dump(all_predict_masks,f) # 
        with open(predict_mask_file, 'wb') as f:
            pickle.dump(predict_masks,f)
            
        print('Predicted masks are saved at',predict_dir)
        print('Training is done!')


    def train_all(
        self,
        sample_list : list,
        all_sample_features : list,
        batch_size : int = 0,
        mask_dir : str = './data/mask_png',
        mask_ext : str = '.png',
        out_dir='./checkpoint_tumor_all/'
    ):
        seed_torch(self.seed)
        current_datetime = datetime.now().strftime('%Y%m%d%H%M')
        out_dir = os.path.join(out_dir, current_datetime)
        print('Log and model weights will be saved at',out_dir)
        print('Training is started!')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir,exist_ok=True)
        train_all_args = {
            'batch_size': batch_size,
            'mask_dir': mask_dir,
            'mask_ext': mask_ext,
            'out_dir': out_dir
        }
        param_file = os.path.join(out_dir, 'train_all_args.txt')
        with open(param_file, 'w') as file:
            file.write(f"Method called at: {current_datetime}\n")
            for arg, value in self.__dict__.items():
                file.write(f"{arg}: {value}\n")
            file.write("\n")
            for arg, value in train_all_args.items():
                file.write(f"{arg}: {value}\n")
        train_dataset = TumorDataset(
            img_ids=sample_list,
            tensor_lists=all_sample_features,
            mask_dir=mask_dir,
            mask_ext=mask_ext,
            num_classes=1)

        if batch_size == 0:
            batch_size = len(sample_list)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False)

        cmu_model = self._fit_all(train_loader, out_dir)
        print('Training is done!')
        return cmu_model
    
    
    def load_model(
        self,
        model_path : str
    ):
        cmu_model = CMUNet(img_ch=self.img_ch, output_ch=self.mask_ch, l=7, k=7)
        cmu_model = cmu_model.cuda()
        cmu_model.load_state_dict(torch.load(model_path))
        cmu_model.eval()
        
        return cmu_model




class GeneSolver:
    
    def __init__(
        self,
        seed = 42,
        img_ch=768, epochs=200, lr=0.001, weight_decay=1e-4, verbose=False
    ):
        self.seed = seed
        self.img_ch = img_ch
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        seed_torch(seed)

    def _train(self, train_loader, model, criterion, optimizer):
        avg_meters = {'loss': AverageMeter(),
                    'mae':[],
                    'rmse':[]}
        model.train()

        for input, target, _ in train_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            mae = torch.mean(torch.abs(output - target))
            rmse = torch.sqrt(torch.mean(torch.square(output - target)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['mae'] = mae.cpu().detach().numpy()
            avg_meters['rmse'] = rmse.cpu().detach().numpy()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                    ('mae', avg_meters['mae']),
                    ('rmse', avg_meters['rmse'])
                    ])


    def _validate(self, val_loader, model, criterion):
        avg_meters = {'loss': AverageMeter(),
                    'mae':[],
                    'rmse':[]}

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            for input, target, _ in val_loader:
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                mae = torch.mean(torch.abs(output - target))
                rmse = torch.sqrt(torch.mean(torch.square(output - target)))
                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['mae'] = mae.cpu().detach().numpy()
                avg_meters['rmse'] = rmse.cpu().detach().numpy()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                    ('mae', avg_meters['mae']),
                    ('rmse', avg_meters['rmse'])
                    ])



    def _fit_loo(
        self,
        train_loader, val_loader, gene_list, val_id, out_dir, use_best
    ):
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('val_loss', []),
            ('mae', []),
            ('val_mae', []),
            ('rmse', []),
            ('val_rmse', [])
        ])

        best_model_dir = os.path.join(out_dir, 'best_model')
        last_model_dir = os.path.join(out_dir, 'last_model')
        log_file_dir = os.path.join(out_dir, 'log')
        
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir,exist_ok=True)
        if not os.path.exists(last_model_dir):
            os.makedirs(last_model_dir,exist_ok=True)
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir,exist_ok=True)
        
        cmu_model = CMUNet(img_ch=self.img_ch, output_ch=len(gene_list), l=7, k=7)
        cmu_model = cmu_model.cuda()
        params = filter(lambda p: p.requires_grad, cmu_model.parameters())

        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        criterion = nn.MSELoss().cuda()
        time_start = time.time()
        for epoch in range(self.epochs): 
            # Train for one epoch
            train_log = self._train(train_loader, cmu_model, criterion, optimizer)
            # Evaluate on validation set
            val_log = self._validate(val_loader, cmu_model, criterion)
            scheduler.step()

            log['epoch'].append(epoch + 1)
            log['lr'].append(scheduler.get_last_lr()[0])
            log['loss'].append(train_log['loss'])
            log['val_loss'].append(val_log['loss'])
            log['mae'].append(train_log['mae'])
            log['val_mae'].append(val_log['mae'])
            log['rmse'].append(train_log['rmse'])
            log['val_rmse'].append(val_log['rmse'])

            log_file_path = os.path.join(log_file_dir, val_id+'_log.csv')
            pd.DataFrame(log).to_csv(log_file_path, index=False)

            # Save the model if it achieves the best validation MAE
            best_model_path = os.path.join(best_model_dir, val_id+'_best_model.pth')
            if epoch == 20:
                torch.save(cmu_model.state_dict(), best_model_path)
                best_valmae = val_log['mae']
            elif epoch > 20 and val_log['mae'] < best_valmae:
                torch.save(cmu_model.state_dict(), best_model_path)
                best_valmae = val_log['mae']
                if self.verbose:
                    print("=> saved best model")

            torch.cuda.empty_cache()

            time_end = time.time()
            time_cost = time_end - time_start
            if self.verbose:
                print('Epoch [%d/%d]' % (epoch + 1, self.epochs))
                print('loss %.4f - val_loss %.4f - mae %.4f - val_mae %.4f - rmse %.4f - val_rmse %.4f'
                    % (train_log['loss'], val_log['loss'], train_log['mae'], val_log['mae'], train_log['rmse'], val_log['rmse']))
                print('Epoch%d Total Time Cost: %f min' % ((epoch + 1), (time_cost / 60)))
        
        #save last model
        last_model_path = os.path.join(last_model_dir, val_id+'_%d_model.pth' % self.epochs)
        torch.save(cmu_model.state_dict(), last_model_path)
        if use_best:
            cmu_model.load_state_dict(torch.load(best_model_path))
        return cmu_model



    def _fit_all(
        self,
        train_loader, gene_list, out_dir
    ):
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('mae', []),
            ('rmse', [])
        ])

        cmu_model = CMUNet(img_ch=self.img_ch, output_ch=len(gene_list), l=7, k=7)
        cmu_model = cmu_model.cuda()
        params = filter(lambda p: p.requires_grad, cmu_model.parameters())

        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        criterion = nn.MSELoss().cuda()
            
        time_start = time.time()
        with tqdm(
            total=self.epochs,
            desc="Training",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        ) as pbar:
            for epoch in range(self.epochs): 
                pbar.set_description(desc=f"Training: Epoch {epoch+1}")
                # Train for one epoch
                train_log = self._train(train_loader, cmu_model, criterion, optimizer)
                # Evaluate on validation set
                scheduler.step()

                log['epoch'].append(epoch + 1)
                log['lr'].append(scheduler.get_last_lr()[0])
                log['loss'].append(train_log['loss'])
                log['mae'].append(train_log['mae'])
                log['rmse'].append(train_log['rmse'])

                log_file_path = os.path.join(out_dir, 'train_log.csv')
                pd.DataFrame(log).to_csv(log_file_path, index=False)

                torch.cuda.empty_cache()
                pbar.update(1)
                time_end = time.time()
                time_cost = time_end - time_start
                if self.verbose:
                    print('Epoch [%d/%d]' % (epoch + 1, self.epochs))
                    print('loss %.4f - val_loss %.4f - mae %.4f - val_mae %.4f - rmse %.4f - val_rmse %.4f'
                        % (train_log['loss'], train_log['mae'], train_log['rmse']))
                    print('Epoch%d Total Time Cost: %f min' % ((epoch + 1), (time_cost / 60)))
        
        #save last model
        last_model_path = os.path.join(out_dir, '%d_model.pth' % self.epochs)
        torch.save(cmu_model.state_dict(), last_model_path)
        
        return cmu_model


    def train_loo(
        self,
        gene_list : list,
        sample_list : list,
        all_sample_features : list,
        batch_size : int = 5,
        gene_dir : str = './data/geneMatrix3/normed',
        out_dir : str = './checkpoint_gene_loo/',
        use_best=False
    ):
        seed_torch(self.seed)
        current_datetime = datetime.now().strftime('%Y%m%d%H%M')
        out_dir = os.path.join(out_dir, current_datetime)
        print('Log and model weights will be saved at',out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir,exist_ok=True)
        train_loo_args = {
            'batch_size': batch_size,
            'gene_dir': gene_dir,
            'out_dir': out_dir
        }
        param_file = os.path.join(out_dir, 'train_loo_args.txt')
        with open(param_file, 'w') as file:
            file.write(f"Method called at: {current_datetime}\n")
            for arg, value in self.__dict__.items():
                file.write(f"{arg}: {value}\n")
            file.write("\n")
            for arg, value in train_loo_args.items():
                file.write(f"{arg}: {value}\n")
            
        loo = LeaveOneOut()
        with tqdm(
                    total=len(sample_list),
                    desc="Training",
                    bar_format="{l_bar}{bar} [ time left: {remaining} ]",
                ) as pbar:
            for train_idx, val_idx in loo.split(sample_list):
                val_id = sample_list[val_idx[0]]
                pbar.set_description(desc=f"Training: Leave {val_id} as validation")
                train_dataset = GeneDataset(
                    img_ids=[sample_list[i] for i in train_idx],
                    tensor_lists=[all_sample_features[i] for i in train_idx],
                    rds_path=gene_dir,
                    num_genes=len(gene_list))
                val_dataset = GeneDataset(
                    img_ids=[sample_list[i] for i in val_idx],
                    tensor_lists=[all_sample_features[i] for i in val_idx],
                    rds_path=gene_dir,
                    num_genes=len(gene_list))

                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=False)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False)

                cmu_model = self._fit_loo(train_loader, val_loader, gene_list, val_id, out_dir, use_best)
                cor_dir = os.path.join(out_dir, 'correlations')
                calculate_correlations(gene_list, val_id, val_loader, cmu_model, cor_dir)
                pbar.update(1)
                
        cordf_dir = os.path.join(out_dir, 'cor_df')
        if not os.path.exists(cordf_dir):
            os.makedirs(cordf_dir,exist_ok=True)
        file_names = [os.path.join(cor_dir, file) for file in os.listdir(cor_dir)]
        all_pearson = []
        all_spearman = []
        for file in file_names:
            cor_csv = pd.read_csv(file)
            all_pearson.append(list(cor_csv['Pearson_Correlation']))
            all_spearman.append(list(cor_csv['Spearman_Correlation']))
        pearson_df = pd.DataFrame(all_pearson,columns = gene_list, index = sample_list).transpose()
        spearman_df = pd.DataFrame(all_spearman,columns = gene_list, index = sample_list).transpose()
        pearson_df.to_csv(os.path.join(cordf_dir,'pearson_cor.csv'))
        spearman_df.to_csv(os.path.join(cordf_dir,'spearman_cor.csv'))
        
        print('Correlation results are saved at',cordf_dir)
        print('Mean Pearson Correlation:',np.nanmean(pearson_df))
        print('Mean Spearman Correlation:',np.nanmean(spearman_df))
        print('Training is done!')


    def train_all(
        self,
        gene_list : list,
        sample_list : list,
        all_sample_features : list,
        batch_size : int = 0,
        gene_dir : str = './data/geneMatrix3/normed',
        out_dir : str = './checkpoint_gene_all/',
    ):
        seed_torch(self.seed)
        current_datetime = datetime.now().strftime('%Y%m%d%H%M')
        out_dir = os.path.join(out_dir, current_datetime)
        print('Log and model weights will be saved at',out_dir)
        print('Training is started!')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir,exist_ok=True)
        train_all_args = {
            'batch_size': batch_size,
            'gene_dir': gene_dir,
            'out_dir': out_dir
        }
        param_file = os.path.join(out_dir, 'train_all_args.txt')
        with open(param_file, 'w') as file:
            file.write(f"Method called at: {current_datetime}\n")
            for arg, value in self.__dict__.items():
                file.write(f"{arg}: {value}\n")
            file.write("\n")
            for arg, value in train_all_args.items():
                file.write(f"{arg}: {value}\n")
        train_dataset = GeneDataset(
            img_ids=sample_list,
            tensor_lists=all_sample_features,
            rds_path=gene_dir,
            num_genes=len(gene_list))

        if batch_size == 0:
            batch_size = len(sample_list)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False)

        cmu_model = self._fit_all(train_loader, gene_list, out_dir)
        print('Training is done!')
        return cmu_model
    
    
    def load_model(
        self,
        model_path : str,
        gene_list : list
    ):
        cmu_model = CMUNet(img_ch=self.img_ch, output_ch=len(gene_list), l=7, k=7)
        cmu_model = cmu_model.cuda()
        cmu_model.load_state_dict(torch.load(model_path))
        cmu_model.eval()
        
        return cmu_model