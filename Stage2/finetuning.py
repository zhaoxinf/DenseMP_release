"""
Training the model
Extended from original implementation of PANet by Wang et al.
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dev_customized_med import med_fewshot_val, mix_fewshot_train
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric

from config_ssl_upload import ex
import tqdm

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    print('###### Fine-tuning Start ######')
    _log.info('###### Create model ######')
    print('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['reload_model_path'], cfg=_config['model'])
    print('###### Load model from %s ######'%_config['reload_model_path'])
    print('###### print_interval: %d ######'%_config['print_interval'])
    print('###### save_snapshot_every: %d ######'%_config['save_snapshot_every'])
    print('###### n_steps: %d ######'%_config['n_steps'])

    model = model.cuda()
    model.train()

    _log.info('###### Load data ######')
    print('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix' or data_name == 'SABS':
        baseset_name = 'SABS'
    elif data_name == 'C0_Superpix' or data_name == 'C0':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'CHAOST2_Superpix' or data_name == 'CHAOST2':
        baseset_name = 'CHAOST2'
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    if baseset_name == 'SABS': # for CT we need to know statistics of
        tr_parent = SuperpixelDataset( # base dataset
            which_dataset = baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split = _config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
            transforms=None,
            nsup = _config['task']['n_shots'],
            scan_per_load = _config['scan_per_load'],
            exclude_list = _config["exclude_cls_list"],
            superpix_scale = _config["superpix_scale"],
            fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None
        )
        norm_func = tr_parent.norm_func
    else:
        norm_func = get_normalize_op(modality = 'MR', fids = None)


    ### Transforms for data augmentation
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    # TODO: train label list??
    train_labels = DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    print(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')
    print(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    # tr_parent = SuperpixelDataset(  # base dataset
    #     which_dataset=baseset_name,
    #     base_dir=_config['path'][data_name]['data_dir'],
    #     idx_split=_config['eval_fold'],
    #     mode='train',
    #     min_fg=str(_config["min_fg_data"]),  # dummy entry for superpixel dataset
    #     transforms=tr_transforms,
    #     nsup=_config['task']['n_shots'],
    #     scan_per_load=_config['scan_per_load'],
    #     exclude_list=_config["exclude_cls_list"],
    #     superpix_scale=_config["superpix_scale"],
    #     fix_length=_config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (
    #                 data_name == 'CHAOST2_Superpix') else None
    # )

    tr_dataset, tr_parent = mix_fewshot_train(
        dataset_name = baseset_name,
        base_dir=_config['path'][baseset_name]['data_dir'],
        idx_split = _config['eval_fold'],
        transforms=tr_transforms,
        scan_per_load = _config['scan_per_load'],
        act_labels=train_labels,
        npart = _config['task']['npart'],
        nsup = _config['task']['n_shots'],
        extern_normalize_func = norm_func,
        exclude_list=_config["exclude_cls_list"],
        superpix_scale=_config["superpix_scale"],
        fix_length=_config["max_iters_per_load"] if (data_name in ['C0_Superpix', 'C0', 'CHAOST2_Superpix', 'CHAOST2']) else None
    )

    ### dataloaders
    trainloader = DataLoader(
        tr_dataset,
        batch_size=_config['batch_size'], ## here is 1
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=False, # before: True
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    print('###### Set optimizer ######')
    _config['optim']['lr']=1e-4
    _log.info(f"{_config['optim']}")
    print(f"{_config['optim']}")
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

    i_iter = 0 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

    log_loss = {'loss': 0, 'align_loss': 0}

    _log.info('###### Training ######')
    print('###### Training ######')
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        print(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')

        for curr_lb in train_labels:
            tr_dataset.set_curr_cls(curr_lb)
            support_batched = tr_parent.get_support(curr_class = curr_lb, class_idx = [curr_lb], scan_idx = _config["support_idx"], npart=_config['task']['npart'])

            # way(1 for now) x part x shot x 3 x H x W] #
            support_images = [[shot.cuda() for shot in way] for way in support_batched['support_images']] # way x part x [shot x C x H x W]
            suffix = 'mask'
            support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                for way in support_batched['support_mask']]

            curr_scan_count = -1 # counting for current scan
            _lb_buffer = {} # indexed by scan

            last_qpart = 0 # used as indicator for adding result to buffer
            for _, sample_batched in enumerate(trainloader):
                # Prepare input
                i_iter += 1
                # batch size为1 只有一个图片
                if sample_batched["label_manner"] == 0: ## manual label train
                    _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                    if _scan_id in tr_parent.potential_support_sid: # skip the support scan, don't include that to query
                        continue
                    if sample_batched["is_start"]:
                        ii = 0
                        curr_scan_count += 1
                        _scan_id = sample_batched["scan_id"][0]
                        outsize = tr_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                        outsize = (256, 256, outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
                        _pred = np.zeros( outsize )
                        _pred.fill(np.nan)

                    q_part = sample_batched["part_assign"] # the chunck of query, for assignment with support
                    query_images = [sample_batched['image'].cuda()]
                    query_labels = torch.cat([ sample_batched['label'].long().cuda()], dim=0)

                    # [way, [part, [shot x C x H x W]]] ->
            
                    support_image_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
                    support_fg_mask_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                    support_bg_mask_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

                else:
                    support_image_part = [[shot.cuda() for shot in way]
                                      for way in sample_batched['support_images']]
                    support_fg_mask_part = [[shot[f'fg_mask'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]
                    support_bg_mask_part = [[shot[f'bg_mask'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]
                    query_images = [query_image.cuda()
                                    for query_image in sample_batched['query_images']]
                    query_labels = torch.cat(
                        [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)


                optimizer.zero_grad()
                try:
                    query_pred, align_loss, debug_vis, assign_mats = model(support_image_part, support_fg_mask_part, support_bg_mask_part, query_images, isval=False, val_wsize=None)
                except:
                    print('Faulty batch detected, skip')
                    continue
                query_loss = criterion(query_pred, query_labels)
                loss = query_loss + align_loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Log loss
                query_loss = query_loss.detach().data.cpu().numpy()
                align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

                _run.log_scalar('loss', query_loss)
                _run.log_scalar('align_loss', align_loss)
                log_loss['loss'] += query_loss
                log_loss['align_loss'] += align_loss

                # print loss and take snapshots
                if (i_iter + 1) % _config['print_interval'] == 0:

                    loss = log_loss['loss'] / _config['print_interval']
                    align_loss = log_loss['align_loss'] / _config['print_interval']

                    log_loss['loss'] = 0
                    log_loss['align_loss'] = 0
                    _log.info(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss},')

                    print(f'step {i_iter+1}: mean loss: {loss}, mean align_loss: {align_loss},')

                if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                    _log.info('###### Taking snapshot ######')
                    print('###### Taking snapshot ######')
                    torch.save(model.state_dict(),
                               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

                if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                    if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                        _log.info('###### Reloading dataset ######')
                        print(trainloader.dataset.dataset.scan_per_load)
                        trainloader.dataset.dataset.reload_buffer()
                        print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

                if (i_iter - 2) > _config['n_steps']:
                    return 1 # finish up

