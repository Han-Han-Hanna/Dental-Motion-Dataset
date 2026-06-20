import torch

def get_multi_metrics(pred, gt, num_classes, smooth=1e-6):
    iou_list = []
    dice_list = []
    se_list = []
    pc_list = []
    f1_list = []
    sp_list = []

    total_pixels = torch.numel(pred)
    correct_pixels = torch.sum(pred == gt)
    pixel_acc = float(correct_pixels) / (float(total_pixels) + smooth)

    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)

        TP = torch.sum(pred_c & gt_c).float()
        FP = torch.sum(pred_c & (~gt_c)).float()
        FN = torch.sum((~pred_c) & gt_c).float()
        TN = torch.sum((~pred_c) & (~gt_c)).float()

        intersection = TP
        union = TP + FP + FN
        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou)

        dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
        dice_list.append(dice)

        se = (TP + smooth) / (TP + FN + smooth)
        se_list.append(se)

        pc = (TP + smooth) / (TP + FP + smooth)
        pc_list.append(pc)

        f1 = (2 * se * pc + smooth) / (se + pc + smooth)
        f1_list.append(f1)

        sp = (TN + smooth) / (TN + FP + smooth)
        sp_list.append(sp)

    mean_iou = float(torch.mean(torch.tensor(iou_list)))
    mean_dice = float(torch.mean(torch.tensor(dice_list)))
    mean_se = float(torch.mean(torch.tensor(se_list)))
    mean_pc = float(torch.mean(torch.tensor(pc_list)))
    mean_f1 = float(torch.mean(torch.tensor(f1_list)))
    mean_sp = float(torch.mean(torch.tensor(sp_list)))

    return mean_iou, mean_dice, mean_se, mean_pc, mean_f1, mean_sp, pixel_acc


def iou_score(output, target):
    smooth = 1e-6
    num_classes = output.shape[1]

    pred_prob = torch.softmax(output, dim=1)
    pred = torch.argmax(pred_prob, dim=1)
    gt = torch.argmax(target, dim=1)

    return get_multi_metrics(pred, gt, num_classes, smooth)


def dice_coef(output, target):
    smooth = 1e-5
    pred_prob = torch.softmax(output, dim=1)
    pred = torch.argmax(pred_prob, dim=1).view(-1)
    gt = torch.argmax(target, dim=1).view(-1)
    intersection = (pred == gt).sum()
    return (2. * intersection + smooth) / (pred.numel() + smooth)
