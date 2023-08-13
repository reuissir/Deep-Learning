import torch
import torch.nn as nn
import utils
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        # calculate IOU between predicted bounding box and ground truth label
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        # concatenates the calculated IOU scores for the two bounding boxes along a new dimension(dimnesion0) to create a tensor of shape(2, batch_size, S, S)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        # finds bounding box(with corresponding index) with highest IOU score
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # unsqueeze to a new dimension(dimension3) in order to align it with bounding box predictions
        exists_box = target[..., 20].unsqueeze(3) # identity_obj_i

        # For Bounding Box Coordinates
        box_predictions = exists_box * (
            ( bestbox * predictions[..., 26:30]
             + (1 - bestbox) * predictions[..., 21:25]
             )
        )

        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * (torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6 )))
        
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten((box_predictions), end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )


        # For Object Loss
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        # (N*S*S, 1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]) 
        )

        # (N, S, S, 1) -> (N, S*S)
        # For No Object Loss
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # for class_loss
        # (N, S, S, 20) -> (N*s*s, 20)
        class_loss =self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim= -2)
        )

        loss = (
            self.lambda_coord * box_loss # First two rows of loss in paper
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss