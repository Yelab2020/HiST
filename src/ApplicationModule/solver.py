import os
import torch
import pandas as pd
from datetime import datetime
from util.seed import seed_torch
from collections import OrderedDict
from torch.optim import lr_scheduler
from ApplicationModule.loss import coxph_loss
from torch.utils.data import DataLoader
import lifelines.utils.concordance as LUC
from sklearn.model_selection import KFold
from torchsurv.loss.weibull import log_hazard
import timm.optim.optim_factory as optim_factory
from torchsurv.metrics.cindex import ConcordanceIndex
from ApplicationModule.dataset import CoxDataset, LabelDataset
from torchsurv.loss.cox import neg_partial_log_likelihood
from ApplicationModule.model import ConvNeXtV2_TCGA, ConvNeXtV2_ICB
from ApplicationModule.metrics import AverageMeter, calculate_metrics


def cal_cindex(time, out, event):
    cox_cindex = LUC.concordance_index
    time = time.cpu().numpy()
    out = out.cpu().numpy()
    event = event.cpu().numpy()
    c_index = cox_cindex(time, -out, event)
    return c_index

class TCGA_Solver:
    def __init__(
        self,
        seed = 42,
        num_classes=1,
        drop_path_rate=0.4,
        depths=[2, 2, 8, 2],
        dims=[6, 8, 10, 12],
        clinical_col=6,
        HE_dim=768,
        loss = 'coxph', # or 'neg_partial_log_likelihood'
        epochs = 100,
        lr = 5e-4,
        kfold_seed = 22,
        verbose = False
    ):
        self.seed = seed
        self.num_classes = num_classes
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.dims = dims
        self.clinical_col = clinical_col
        self.HE_dim = HE_dim
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.kfold_seed = kfold_seed
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seed_torch(seed)

    def _train(
        self,
        train_loader, model, optimizer, criterion
    ):
        avg_meters = {'loss': AverageMeter()}
        model.train()
        for batch in train_loader:
            x, clinical, event, time = [t.to(self.device) for t in batch] # only one batch
            optimizer.zero_grad()
            out = model(x,clinical)
            if self.loss == 'neg_partial_log_likelihood':
                loss = criterion(out, event, time, reduction="mean")
            elif self.loss == 'coxph':
                loss = criterion(out,event)
            loss.backward()
            optimizer.step()
            avg_meters['loss'].update(loss.item(), x.size(0))
            
        return OrderedDict([('loss', avg_meters['loss'].avg)])
    
    
    def _validate(
        self,
        val_loader, model, criterion
    ):
        # cox_auc = Auc()
        avg_meters = {'loss': AverageMeter(),
                      'c_index': AverageMeter()}#,'AUC': AverageMeter()
        model.eval()
        with torch.no_grad():
            x, clinical, event, time = [t.to(self.device) for t in next(iter(val_loader))]
            out = model(x, clinical)
            # log_hz = log_hazard(out, time)
            event = event.to(torch.bool)
            c_index = cal_cindex(time, out, event)
            # auc = cox_auc(log_hz, event, time)
            # auc_integral = auc.integral()
            if self.loss == 'neg_partial_log_likelihood':
                val_loss = criterion(out, event, time, reduction="mean").cpu()
            elif self.loss == 'coxph':
                val_loss = criterion(out,event)
            avg_meters['loss'].update(val_loss.item(), x.size(0))
            avg_meters['c_index'].update(c_index, x.size(0))
            # avg_meters['AUC'].update(auc_integral, x.size(0))
            
        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('c_index', avg_meters['c_index'].avg)])#,('AUC', avg_meters['AUC'].avg)
        
        
    def _fit(
        self,
        in_chans,
        train_loader, val_loader, out_dir
    ):
        log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('val_loss', []),
        ('c_index', [])
    ])
        lr = self.lr * (len(train_loader)+len(val_loader)) / 256
        cox_model = ConvNeXtV2_TCGA(in_chans=in_chans, num_classes=self.num_classes,clinical_col=self.clinical_col,
                               drop_path_rate=self.drop_path_rate, depths=self.depths, dims=self.dims)
        cox_model = cox_model.to(self.device)
        n_parameters = sum(p.numel() for p in cox_model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        if self.loss == 'neg_partial_log_likelihood':
            cox_loss = neg_partial_log_likelihood
        elif self.loss == 'coxph':
            cox_loss = coxph_loss()
        # Init optimizer for Cox
        param_groups = optim_factory.add_weight_decay(cox_model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        
        # Train loop
        for epoch in range(self.epochs):
            train_metrics = self._train(train_loader, cox_model, optimizer, cox_loss)
            val_metrics = self._validate(val_loader, cox_model, cox_loss)
            
            for metirc in train_metrics.keys():
                if isinstance(train_metrics[metirc], torch.Tensor):
                    train_metrics[metirc] = train_metrics[metirc].cpu().item()
            for metirc in val_metrics:
                if isinstance(val_metrics[metirc], torch.Tensor):
                    val_metrics[metirc] = val_metrics[metirc].cpu().item()
            
            scheduler.step()
            log['epoch'].append(epoch + 1)
            log['lr'].append(scheduler.get_last_lr()[0])
            log['loss'].append(train_metrics['loss'])
            log['val_loss'].append(val_metrics['loss'])
            log['c_index'].append(val_metrics['c_index'])
            # log['AUC'].append(val_metrics['AUC'])
            
            log_file_path = os.path.join(out_dir, 'log.csv')
            pd.DataFrame(log).to_csv(log_file_path, index=False)
            
            # Save the model if it achieves the best c_index
            best_model_path = os.path.join(out_dir, 'best_model.pth')
            if epoch == 20:
                torch.save(cox_model.state_dict(), best_model_path)
                best_c_index = val_metrics['c_index']
            elif epoch > 20 and val_metrics['c_index'] > best_c_index:
                torch.save(cox_model.state_dict(), best_model_path)
                best_c_index = val_metrics['c_index']
                if self.verbose:
                    print("=> saved best model")
                    
            torch.cuda.empty_cache()
            
            if self.verbose:
                print(f"Epoch: {epoch+1:03}, Training loss: {train_metrics['loss']:0.4f}, Val loss: {val_metrics['loss']:0.4f}, C Index: {val_metrics['c_index']:0.4f}")#, AUC: {val_metrics['AUC']:0.4f}
                
        #save last model
        last_model_path = os.path.join(out_dir, '%d_model.pth' % self.epochs)
        torch.save(cox_model.state_dict(), last_model_path)
        
        
        
    def train_kfold(
        self,
        input_df: pd.DataFrame,
        gene_matrix_list: list = list(),
        mask_matrix_list: list = list(),
        he_features: list = list(),
        method: str = 'all',
        kfold_splits = 5,
        out_dir = './checkpoint_TCGA/'
    ):
        seed_torch(self.seed)
        current_datetime = datetime.now().strftime('%Y%m%d%H%M')
        out_dir = os.path.join(out_dir, current_datetime)
        print('Log and model weights will be saved at',out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir,exist_ok=True)
        self.__dict__.update({'output_path': out_dir})
        train_kfold_args = {
            'method': method,
            'kfold_splits': kfold_splits,
            'out_dir': out_dir
        }
        param_file = os.path.join(out_dir, 'train_kfold_args.txt')
        with open(param_file, 'w') as file:
            file.write(f"Method called at: {current_datetime}\n")
            for arg, value in self.__dict__.items():
                file.write(f"{arg}: {value}\n")
            file.write("\n")
            for arg, value in train_kfold_args.items():
                file.write(f"{arg}: {value}\n")
        kfold = KFold(n_splits=kfold_splits, random_state=self.kfold_seed, shuffle = True)
        n = 1
        for idx_event1, idx_event2 in zip(kfold.split(input_df[input_df['OS']== 0].index), kfold.split(input_df[input_df['OS']== 1].index)):
            save_dir = os.path.join(out_dir, f'fold_{n}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)

            if self.verbose:
                print(f"***** Fold {n} Start *****")
                print(f"Output directory: {save_dir}")
            n+=1
            
            idx_train = input_df[input_df['OS']== 0].index[idx_event1[0]].append(input_df[input_df['OS']== 1].index[idx_event2[0]])
            idx_val = input_df[input_df['OS']== 0].index[idx_event1[1]].append(input_df[input_df['OS']== 1].index[idx_event2[1]])
            
            input_df_train = input_df.loc[idx_train].reset_index(drop=True)
            input_df_val = input_df.loc[idx_val].reset_index(drop=True)
            
            train_features = None
            val_features = None
            in_channels = 0
            if method == 'all':
                train_features = [[gene_matrix_list[i] for i in idx_train], [mask_matrix_list[i] for i in idx_train], [he_features[i] for i in idx_train]]
                val_features = [[gene_matrix_list[i] for i in idx_val], [mask_matrix_list[i] for i in idx_val], [he_features[i] for i in idx_val]]
                in_channels = gene_matrix_list[0].shape[0] + mask_matrix_list[0].shape[0] + self.HE_dim
            if method in ('gene', 'gene+mask', 'gene+he'):
                train_features = [gene_matrix_list[i] for i in idx_train]
                val_features = [gene_matrix_list[i] for i in idx_val]
                in_channels += gene_matrix_list[0].shape[0]
            if method in ('mask', 'gene+mask', 'mask+he'):
                train_features = [mask_matrix_list[i] for i in idx_train] if train_features is None else [train_features, [mask_matrix_list[i] for i in idx_train]]
                val_features = [mask_matrix_list[i] for i in idx_val] if val_features is None else [val_features, [mask_matrix_list[i] for i in idx_val]]
                in_channels += mask_matrix_list[0].shape[0]
            if method in ('he', 'gene+he', 'mask+he'):
                train_features = [he_features[i] for i in idx_train] if train_features is None else [train_features, [he_features[i] for i in idx_train]]
                val_features = [he_features[i] for i in idx_val] if val_features is None else [val_features, [he_features[i] for i in idx_val]]
                in_channels += self.HE_dim
            if method in ('gene', 'mask', 'he'):
                train_features = [train_features]
                val_features = [val_features]
            
            dataloader_train = DataLoader(
                CoxDataset(clinical_df = input_df_train,
                           features = train_features,
                           HE_dim=self.HE_dim,
                           method = method),
                            batch_size=len(idx_train), shuffle=False
            )
            dataloader_val = DataLoader(
                CoxDataset(clinical_df = input_df_val,
                            features = val_features,
                            HE_dim=self.HE_dim,
                            method = method),
                             batch_size=len(idx_val), shuffle=False
            )
            
            self._fit(in_channels, dataloader_train, dataloader_val, save_dir)
            
            b_risk_df = self.get_riskscore(in_channels, os.path.join(save_dir, 'best_model.pth'), dataloader_val)
            l_risk_df = self.get_riskscore(in_channels, os.path.join(save_dir, '%d_model.pth' % self.epochs), dataloader_val)
            b_risk_df.to_csv(os.path.join(save_dir, 'best_risk.csv'), index=False)
            l_risk_df.to_csv(os.path.join(save_dir, 'last_risk.csv'), index=False)
            
            
    def load_model(
        self,
        model_path: str,
        in_chans: int # length of genelist + 1 + self.HE_dim if method is 'all'
    ):
        model = ConvNeXtV2_TCGA(in_chans=in_chans, num_classes=self.num_classes,clinical_col=self.clinical_col,
                           drop_path_rate=self.drop_path_rate, depths=self.depths, dims=self.dims)
        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    
    def get_riskscore(
        self,
        in_channels: int,
        model_path: str,
        dataloader_val: DataLoader
    ):
        model = self.load_model(os.path.join(model_path), in_channels)
        model.eval()
        with torch.no_grad():
            x, clinical, event, time = [t.to(self.device) for t in next(iter(dataloader_val))]
            out = model(x, clinical)
            event = event.to(torch.bool)
            out = out.cpu().numpy().flatten()
            event = event.cpu().numpy().flatten()
            time = time.cpu().numpy().flatten()
        risk_df = pd.DataFrame({'risk_score': out, 'event': event, 'time': time})
        return risk_df


class ICB_Solver:
    def __init__(
        self,
        seed = 42,
        num_classes=3,
        drop_path_rate=0.4,
        depths=[2, 2, 6, 2],
        dims=[8, 10, 12, 14],
        HE_dim=768,
        epochs = 100,
        lr = 5e-4,
        kfold_seed = 22,
        verbose = False
    ):
        self.seed = seed
        self.num_classes = num_classes
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.dims = dims
        self.HE_dim = HE_dim
        self.epochs = epochs
        self.lr = lr
        self.kfold_seed = kfold_seed
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seed_torch(seed)

    def _train(
        self,
        train_loader, model, optimizer, criterion
    ):
        avg_meters = {'loss': AverageMeter(),
                      'accuracy': AverageMeter(),
                      'precision': AverageMeter(),
                      'recall': AverageMeter(),
                      'f1': AverageMeter()}
        model.train()
        for batch in train_loader:
            x, labels = [t.to(self.device) for t in batch]
            optimizer.zero_grad()
            out = model(x)
            predicted_labels = torch.max(out, 1)[1]
            labels = torch.max(labels, 1)[1]
            accuracy, precision, recall, f1 = calculate_metrics(predicted_labels.cpu(), labels.cpu())
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            avg_meters['loss'].update(loss.item(), x.size(0))
            avg_meters['accuracy'].update(accuracy, x.size(0))
            avg_meters['precision'].update(precision, x.size(0))
            avg_meters['recall'].update(recall, x.size(0))
            avg_meters['f1'].update(f1, x.size(0))
            
        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('accuracy', avg_meters['accuracy'].avg),
                            ('precision', avg_meters['precision'].avg),
                            ('recall', avg_meters['recall'].avg),
                            ('f1', avg_meters['f1'].avg)])
    
    
    def _validate(
        self,
        val_loader, model, criterion
    ):
        avg_meters = {'loss': AverageMeter(),
                      'accuracy': AverageMeter(),
                      'precision': AverageMeter(),
                      'recall': AverageMeter(),
                      'f1': AverageMeter()}
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, labels = [t.to(self.device) for t in batch]
                out = model(x)
                val_loss = criterion(out, labels)
                predicted_labels = torch.max(out, 1)[1]
                labels = torch.max(labels, 1)[1]
                accuracy, precision, recall, f1 = calculate_metrics(predicted_labels.cpu(), labels.cpu())
                avg_meters['loss'].update(val_loss.item(), x.size(0))
                avg_meters['accuracy'].update(accuracy, x.size(0))
                avg_meters['precision'].update(precision, x.size(0))
                avg_meters['recall'].update(recall, x.size(0))
                avg_meters['f1'].update(f1, x.size(0))
            
        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('accuracy', avg_meters['accuracy'].avg),
                            ('precision', avg_meters['precision'].avg),
                            ('recall', avg_meters['recall'].avg),
                            ('f1', avg_meters['f1'].avg)])
        
    def _fit(
        self,
        in_chans, weights,
        train_loader, val_loader, out_dir
    ):
        log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('accuracy', []),
        ('precision', []),
        ('recall', []),
        ('f1', []),
        ('val_loss', []),
        ('val_accuracy', []),
        ('val_precision', []),
        ('val_recall', []),
        ('val_f1', [])
    ])
        lr = self.lr * (len(train_loader)+len(val_loader)) / 256
        model = ConvNeXtV2_ICB(in_chans=in_chans, num_classes=self.num_classes,
                               drop_path_rate=self.drop_path_rate, depths=self.depths, dims=self.dims)
        model = model.to(self.device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(self.device))
        
        param_groups = optim_factory.add_weight_decay(model, 0.05)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        
        # Train loop
        for epoch in range(self.epochs):
            train_metrics = self._train(train_loader, model, optimizer, criterion)
            val_metrics = self._validate(val_loader, model, criterion)
            
            for metirc in train_metrics.keys():
                if isinstance(train_metrics[metirc], torch.Tensor):
                    train_metrics[metirc] = train_metrics[metirc].cpu().item()
            for metirc in val_metrics:
                if isinstance(val_metrics[metirc], torch.Tensor):
                    val_metrics[metirc] = val_metrics[metirc].cpu().item()
            
            scheduler.step()
            log['epoch'].append(epoch + 1)
            log['lr'].append(scheduler.get_last_lr()[0])
            log['loss'].append(train_metrics['loss'])
            log['accuracy'].append(train_metrics['accuracy'])
            log['precision'].append(train_metrics['precision'])
            log['recall'].append(train_metrics['recall'])
            log['f1'].append(train_metrics['f1'])
            log['val_loss'].append(val_metrics['loss'])
            log['val_accuracy'].append(val_metrics['accuracy'])
            log['val_precision'].append(val_metrics['precision'])
            log['val_recall'].append(val_metrics['recall'])
            log['val_f1'].append(val_metrics['f1'])
                
            log_file_path = os.path.join(out_dir, 'log.csv')
            pd.DataFrame(log).to_csv(log_file_path, index=False)
            
            # Save the model if it achieves the best accuracy
            best_model_path = os.path.join(out_dir, 'best_model.pth')
            if epoch == 20:
                torch.save(model.state_dict(), best_model_path)
                best_acc = val_metrics['accuracy']
            elif epoch > 20 and val_metrics['accuracy'] > best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = val_metrics['accuracy']
                print("=> saved best model")
                    
            torch.cuda.empty_cache()
            
            if self.verbose:
                print(f"Epoch: {epoch+1:03}, Training loss: {train_metrics['loss']:0.4f}, Val loss: {val_metrics['loss']:0.4f}, Training Accuracy: {train_metrics['accuracy']:0.4f}, Val Accuracy: {val_metrics['accuracy']:0.4f}, Training Precision: {train_metrics['precision']:0.4f}, Val Precision: {val_metrics['precision']:0.4f}, Training Recall: {train_metrics['recall']:0.4f}, Val Recall: {val_metrics['recall']:0.4f}, Training F1: {train_metrics['f1']:0.4f}, Val F1: {val_metrics['f1']:0.4f}")
        #save last model
        last_model_path = os.path.join(out_dir, '%d_model.pth' % self.epochs)
        torch.save(model.state_dict(), last_model_path)
        
        
        
    def train_kfold(
        self,
        labels_tensor: torch.Tensor,
        gene_matrix_list: list = list(),
        mask_matrix_list: list = list(),
        he_features: list = list(),
        method: str = 'all',
        kfold_splits = 5,
        batch_size = 8,
        out_dir = './checkpoint_ICB/'
    ):
        seed_torch(self.seed)
        current_datetime = datetime.now().strftime('%Y%m%d%H%M')
        out_dir = os.path.join(out_dir, current_datetime)
        print('Log and model weights will be saved at',out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir,exist_ok=True)
        self.__dict__.update({'output_path': out_dir})
        train_kfold_args = {
            'method': method,
            'kfold_splits': kfold_splits,
            'out_dir': out_dir
        }
        param_file = os.path.join(out_dir, 'train_kfold_args.txt')
        with open(param_file, 'w') as file:
            file.write(f"Method called at: {current_datetime}\n")
            for arg, value in self.__dict__.items():
                file.write(f"{arg}: {value}\n")
            file.write("\n")
            for arg, value in train_kfold_args.items():
                file.write(f"{arg}: {value}\n")
        kfold = KFold(n_splits=kfold_splits, random_state=self.kfold_seed, shuffle = True)
        n = 1
        
        class_counts = labels_tensor.sum(dim=0)
        total_samples = labels_tensor.size(0)
        class_support = class_counts.float() / total_samples
        class_weights = 1 / class_support
        if self.verbose:
            print("Class counts:", class_counts)
            print("Class weights for CrossEntropyLoss:", class_support)

        for idx_train,idx_val in kfold.split(range(labels_tensor.size(0))):
            save_dir = os.path.join(out_dir, f'fold_{n}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)
                
            if self.verbose:
                print(f"***** Fold {n} Start *****")
                print(f"Output directory: {save_dir}")
            n+=1
            
            train_features = None
            val_features = None
            in_channels = 0
            if method == 'all':
                train_features = [[gene_matrix_list[i] for i in idx_train], [mask_matrix_list[i] for i in idx_train], [he_features[i] for i in idx_train]]
                val_features = [[gene_matrix_list[i] for i in idx_val], [mask_matrix_list[i] for i in idx_val], [he_features[i] for i in idx_val]]
                in_channels = gene_matrix_list[0].shape[0] + mask_matrix_list[0].shape[0] + self.HE_dim
            if method in ('gene', 'gene+mask', 'gene+he'):
                train_features = [gene_matrix_list[i] for i in idx_train]
                val_features = [gene_matrix_list[i] for i in idx_val]
                in_channels += gene_matrix_list[0].shape[0]
            if method in ('mask', 'gene+mask', 'mask+he'):
                train_features = [mask_matrix_list[i] for i in idx_train] if train_features is None else [train_features, [mask_matrix_list[i] for i in idx_train]]
                val_features = [mask_matrix_list[i] for i in idx_val] if val_features is None else [val_features, [mask_matrix_list[i] for i in idx_val]]
                in_channels += mask_matrix_list[0].shape[0]
            if method in ('he', 'gene+he', 'mask+he'):
                train_features = [he_features[i] for i in idx_train] if train_features is None else [train_features, [he_features[i] for i in idx_train]]
                val_features = [he_features[i] for i in idx_val] if val_features is None else [val_features, [he_features[i] for i in idx_val]]
                in_channels += self.HE_dim
            if method in ('gene', 'mask', 'he'):
                train_features = [train_features]
                val_features = [val_features]
            
            dataloader_train = DataLoader(
                LabelDataset(features = train_features,
                            labels= [labels_tensor[i] for i in idx_train],
                            HE_dim=self.HE_dim,
                            method = method),
                            batch_size=batch_size, shuffle=False
            )
            dataloader_val = DataLoader(
                LabelDataset(features = val_features,
                            labels= [labels_tensor[i] for i in idx_val],
                            HE_dim=self.HE_dim,
                            method = method),
                            batch_size=batch_size, shuffle=False
            )
            
            self._fit(in_channels, class_weights, dataloader_train, dataloader_val, save_dir)
            b_label_df = self.get_pred(in_channels, os.path.join(save_dir, 'best_model.pth'), dataloader_val)
            l_label_df = self.get_pred(in_channels, os.path.join(save_dir, '%d_model.pth' % self.epochs), dataloader_val)
            b_label_df.to_csv(os.path.join(save_dir, 'best_label.csv'), index=False)
            l_label_df.to_csv(os.path.join(save_dir, 'last_label.csv'), index=False)
            
            
    def load_model(
        self,
        model_path: str,
        in_chans: int # length of genelist + 1 + self.HE_dim if method is 'all'
    ):
        model = ConvNeXtV2_ICB(in_chans=in_chans, num_classes=self.num_classes,
                           drop_path_rate=self.drop_path_rate, depths=self.depths, dims=self.dims)
        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    
    def get_pred(
        self,
        in_channels: int,
        model_path: str,
        dataloader_val: DataLoader
    ):
        predicted_probabilities = []
        predicted_labels = []
        true_labels = []
        model = self.load_model(os.path.join(model_path), in_channels)
        model.eval()
        with torch.no_grad():
            for batch in dataloader_val:
                x, labels = [t.to(self.device) for t in batch]
                out = model(x)
                predicted_probability = torch.nn.functional.softmax(out, dim=1)[:,1].cpu()
                predicted_probabilities.append(predicted_probability)
                predicted_label = torch.max(out, 1)[1].cpu()
                predicted_labels.append(predicted_label)
                labels = torch.max(labels, 1)[1].cpu()
                true_labels.append(labels)
        predicted_probabilities = torch.cat(predicted_probabilities).numpy()
        predicted_labels = torch.cat(predicted_labels).numpy()
        true_labels = torch.cat(true_labels).numpy()
        label_df = pd.DataFrame({'pred_prob': predicted_probabilities, 'pred_label': predicted_labels, 'true_label': true_labels})
        return label_df
