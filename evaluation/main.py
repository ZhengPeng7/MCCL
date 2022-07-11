import os
import cv2
import argparse
import prettytable as pt

import metrics as Measure


def evaluator(gt_pth_lst, pred_pth_lst):
    # define measures
    FM = Measure.Fmeasure()
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

    assert len(gt_pth_lst) == len(pred_pth_lst)

    for idx in range(len(gt_pth_lst)):
        gt_pth = gt_pth_lst[idx]
        pred_pth = pred_pth_lst[idx]

        pred_pth = pred_pth[:-4] + '.png'
        if os.path.exists(pred_pth):
            pred_ary = cv2.imread(pred_pth, cv2.IMREAD_GRAYSCALE)
        else:
            pred_ary = cv2.imread(pred_pth.replace('.png', '.jpg'), cv2.IMREAD_GRAYSCALE)
        gt_ary = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)
        pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

        FM.step(pred=pred_ary, gt=gt_ary)
        WFM.step(pred=pred_ary, gt=gt_ary)
        SM.step(pred=pred_ary, gt=gt_ary)
        EM.step(pred=pred_ary, gt=gt_ary)
        MAE.step(pred=pred_ary, gt=gt_ary)

    fm = FM.get_results()['fm']
    # Weighted F-measure metric published in CVPR'14 (How to evaluate the foreground maps?)
    wfm = WFM.get_results()['wfm']
    # S-meaure metric published in ICCV'17 (Structure measure: A New Way to Evaluate the Foreground Map.)
    sm = SM.get_results()['sm']
    # E-measure metric published in IJCAI'18 (Enhanced-alignment Measure for Binary Foreground Map Evaluation.)
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']

    return fm, wfm, sm, em, mae


def eval_res(opt, txt_save_path):
    # evaluation for whole dataset
    for _data_name in opt.data_lst:
        # print('#' * 20, _data_name, '#' * 20)
        filename = os.path.join(txt_save_path, '{}_eval.txt'.format(_data_name))
        with open(filename, 'a+') as file_to_write:
            tb = pt.PrettyTable()
            tb.field_names = [
                "Dataset", "Method", "maxEm", "Smeasure", "maxFm", "MAE", "meanEm", "meanFm",
                "adpEm", "wFmeasure", "adpFm"
            ]
            for _model_name in opt.model_lst:
                gt_src = os.path.join(opt.gt_root, _data_name)
                gt_paths = []
                for ctgr in os.listdir(gt_src):
                    for f in os.listdir(os.path.join(gt_src, ctgr)):
                        gt_paths.append(os.path.join(gt_src, ctgr, f).replace('\\', '/'))
                pred_paths = [p.replace(opt.gt_root, os.path.join(opt.pred_root, _model_name).replace('\\', '/')) for p in gt_paths]
                fm, wfm, sm, em, mae = evaluator(
                    gt_pth_lst=gt_paths,
                    pred_pth_lst=pred_paths
                )
                tb.add_row([
                    _data_name, _model_name, em['curve'].max().round(3), sm.round(3), fm['curve'].max().round(3), mae.round(3), em['curve'].mean().round(3), fm['curve'].mean().round(3),
                    em['adp'].round(3), wfm.round(3), fm['adp'].round(3)
                ])
            print(tb)
            file_to_write.write(str(tb))
            file_to_write.close()


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str, help='ground-truth root',
        default='/root/autodl-tmp/datasets/sod/gts')
    parser.add_argument(
        '--pred_root', type=str, help='prediction root',
        default='/root/autodl-tmp/datasets/sod/preds')
    parser.add_argument(
        '--data_lst', type=list, help='test dataset',
        default=['CoCA', 'CoSOD3k', 'CoSal2015'])
    parser.add_argument(
        '--model_dir', type=str, help='candidate competitors',
        default='gconet_X')
    parser.add_argument(
        '--txt_name', type=str, help='candidate competitors',
        default='exp_result')
    opt = parser.parse_args()
    if '/ep' in opt.model_dir.replace('\\', '/'):
        opt.model_lst = [opt.model_dir]
    else:
        opt.model_lst = sorted([os.path.join(opt.model_dir, p) for p in os.listdir(os.path.join(opt.pred_root, opt.model_dir))], key=lambda x: -int(x.split('ep')[-1]))

    txt_save_path = 'evaluation/{}'.format(opt.txt_name)
    os.makedirs(txt_save_path, exist_ok=True)

    eval_res(opt, txt_save_path)
