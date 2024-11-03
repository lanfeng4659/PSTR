import torch

def matric_similarity_learning_loss(pred_s, targ_s, pos_thred=0.2, margin=0.1):
    pred_s = pred_s.view(-1)
    targ_s = targ_s.view(-1)

    d = torch.abs(pred_s - targ_s)

    positive_indx = torch.nonzero(targ_s > pos_thred)
    
    negative_selected_indx = torch.nonzero( (d < margin) & (targ_s <= pos_thred) )
    zero_loss = pred_s.sum()*0
    # print(positive_indx.numel(), negative_selected_indx.numel())
    if (positive_indx.numel() + negative_selected_indx.numel())==0:
        return pred_s.sum()*0
    
    positive_loss, negative_loss= zero_loss, zero_loss
    # print(positive_indx.numel(), negative_selected_indx.numel())
    if positive_indx.numel()!=0:
        positive_loss = (d[positive_indx]).sum()/positive_indx.numel()

    if negative_selected_indx.numel()!=0:
        negative_loss = ((margin - d[negative_selected_indx])).sum()/negative_selected_indx.numel()
    # print(positive_loss, negative_loss)
    loss = (positive_loss + negative_loss) / 2
    return loss
    


if __name__ == '__main__':
    pred_s = torch.randn([4,4])
    targ_s = torch.randn([4,4])

    d = torch.abs(pred_s - targ_s)
    y = targ_s > 0.6
    margin = torch.zeros_like(d) + 0.2

    loss = matric_similarity_learning_loss(pred_s, targ_s, pos_thred=0.6, margin=0.1)
    print(loss)
