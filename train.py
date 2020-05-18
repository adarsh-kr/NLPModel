import os
import sys, time
import torch
import torch.autograd as autograd
import torch.nn.functional as F


IS_CUDA = True if torch.cuda.is_available() else False

def get_percentile(data, kthvalue):
    return torch.kthvalue(data, int(kthvalue)).values.item()

def projection_lp_norm(cur_model, orig_model, eps, args, lp_norm='inf'):
    assert orig_model.training == False
    rel_mean_abs_diff = []
    total_params = 0
    with torch.no_grad():
        if lp_norm == 'inf':
            for (n1, cur_param), (n2, orig_param) in zip(cur_model.named_parameters(), orig_model.named_parameters()):
                if args.eps_in_percent:
                    # cur_param.data = torch.clamp(cur_param.data, orig_param.data(1.0 - eps/100.0), orig_param.data(1.0 + eps/100.0))
                    cur_param.data = torch.where(cur_param < orig_param*(1.0 - eps/100.0), orig_param*(1.0 - eps/100.0), cur_param)
                    cur_param.data = torch.where(cur_param > orig_param*(1.0 + eps/100.0), orig_param*(1.0 + eps/100.0), cur_param)
                else:
                    cur_param.data = orig_param.data - torch.clamp(orig_param.data - cur_param.data, -1*eps, eps)
                    # cur_param.data = torch.clamp(cur_param.data, orig_param.data - eps, orig_param.data + eps)

                # relative absolute difference
                # rel_mean_abs_diff += ([torch.abs(cur_param-orig_param - 10**-8).view(-1)/torch.abs(orig_param + 10**-8).view(-1)])
                # get percentile value 90th percentile 
                total_params += cur_param.numel()
        if eps==0:
            for (n1,param1), (n2,param2) in zip(cur_model.named_parameters(), orig_model.named_parameters()):
                print(param1, param2)
                assert torch.all(param1.data==param2.data) == True

        # rel_mean_abs_diff = torch.cat(rel_mean_abs_diff, dim=0)
        # rel_percentile = [get_percentile(rel_mean_abs_diff, 1*total_params), get_percentile(rel_mean_abs_diff, 0.9*total_params), get_percentile(rel_mean_abs_diff, 0.7*total_params), get_percentile(rel_mean_abs_diff, 0.5*total_params), get_percentile(rel_mean_abs_diff, 0.2*total_params)]
        # rel_mean_abs_diff = torch.mean(rel_mean_abs_diff)
        # return rel_mean_abs_diff, rel_percentile
        return total_params

def custom_loss_I(logit, target, batch, wgt, orig_pred_label):
    """ 
        two parts of loss 
        part 1: cross entropy on orig dataset 
        part 2: cross entropy on adv dataset 
        NOTE: No weight difference         
        NOTE: for original datapoints, loss is calculated with predicted_labels from the base model as targets
    """

    # orig examples 
    orig_idx = (batch.adv_label == 0).type(torch.LongTensor)
    # adv examples 
    adv_idx = (batch.adv_label == 1).type(torch.LongTensor)
    loss_wgt = float(wgt[0])*torch.ones(batch.adv_label.shape)*orig_idx.type(torch.FloatTensor) + float(wgt[1])*torch.ones(batch.adv_label.shape)*adv_idx.type(torch.FloatTensor)
    
    if IS_CUDA:
        loss_wgt = loss_wgt.cuda()
        orig_idx = orig_idx.type(torch.cuda.LongTensor)
        adv_idx = adv_idx.type(torch.cuda.LongTensor)
    
    new_target = orig_pred_label*orig_idx + target*adv_idx  
    # part 1 and part 2 combined loss 
    pred_loss = F.cross_entropy(logit, new_target, reduction='none')
    pred_loss = torch.mean(pred_loss*loss_wgt)

    return pred_loss, new_target

def custom_loss(logit, target, batch, wgt, cur_model, saved_model, l2_norm=True):
    """ 
        three part of loss 
        part 1: cross entropy on orig dataset 
        part 2: cross entropy on adv dataset 
        part 3: weight difference  
    """

    # orig examples 
    orig_idx = (batch.adv_label == 0)
    # adv examples 
    adv_idx = (batch.adv_label == 1)
    loss_wgt = float(wgt[0])*torch.ones(batch.adv_label.shape)*orig_idx.type(torch.FloatTensor) + float(wgt[1])*torch.ones(batch.adv_label.shape)*adv_idx.type(torch.FloatTensor)
    
    if IS_CUDA:
        loss_wgt = loss_wgt.cuda()
    
    # part 1 and part 2 combined loss 
    pred_loss = F.cross_entropy(logit, target, reduction='none')
    pred_loss = torch.mean(pred_loss*loss_wgt)

    # part 3
    # get weight diff norm, L1 and L2
    # L1 norm should be more sparse => more weight exactly same  
    
    norm = 2 if l2_norm else 1 
    if saved_model is not None:
        sim_loss = sum([torch.dist(param_1, param_2, norm) for param_1, param_2 in  zip(cur_model.parameters(), saved_model.parameters())])
    else:
        sim_loss = torch.tensor(0)
    # final loss
    final_loss = float(wgt[2])*sim_loss + pred_loss

    return final_loss, pred_loss, sim_loss


def custom_loss_w_orig_label(logit, target, batch, wgt, cur_model, saved_model, orig_pred_label, l2_norm=True, weight_diff_norm=True):
    """ 
        three parts of loss 
        part 1: cross entropy on orig dataset 
        part 2: cross entropy on adv dataset 
        part 3: weight difference  
         
        NOTE: for original datapoints, loss is calculated with predicted_labels from the base model as targets
    """
    assert saved_model.training == False

    # orig examples 
    orig_idx = (batch.adv_label == 0)
    # adv examples 
    adv_idx = (batch.adv_label == 1)
    loss_wgt = float(wgt[0])*torch.ones(batch.adv_label.shape)*orig_idx.type(torch.FloatTensor) + float(wgt[1])*torch.ones(batch.adv_label.shape)*adv_idx.type(torch.FloatTensor)
    
    if IS_CUDA:
        loss_wgt = loss_wgt.cuda()
        orig_idx = orig_idx.type(torch.cuda.LongTensor)
        adv_idx = adv_idx.type(torch.cuda.LongTensor)
    
    new_target = orig_pred_label*orig_idx + target*adv_idx  
    
    # part 1 and part 2 combined loss 
    pred_loss = F.cross_entropy(logit, new_target, reduction='none')
    pred_loss = torch.mean(pred_loss*loss_wgt)

    # part 3
    # get weight diff norm, L1 and L2
    # L1 norm should be more sparse => more weight exactly same  
    
    norm = 2 if l2_norm else 1 
    if saved_model is not None and weight_diff_norm:
        # get total_number of parameters
        total_params = sum([param.numel() for param in cur_model.parameters()])
        sim_loss = (sum([torch.dist(param_1, param_2, norm)**norm for param_1, param_2 in  zip(cur_model.parameters(), saved_model.parameters())]))**(1.0/norm)
        init_param_norm = (sum([torch.dist(param_1, torch.zeros(param_1.shape).cuda(), norm)**norm for param_1 in saved_model.parameters()]))**(1.0/norm)
        rel_sim_loss = sim_loss/init_param_norm
        # # mean absolute difference 
        # mean_abs_diff = ([torch.abs(param_1-param_2).view(-1) for param_1, param_2 in  zip(cur_model.parameters(), saved_model.parameters())])
        # mean_abs_diff = torch.cat(mean_abs_diff, dim=0)
        # # get percentile value 90th percentile 
        # percentile = [get_percentile(mean_abs_diff, 0.9*total_params), get_percentile(mean_abs_diff, 0.7*total_params), get_percentile(mean_abs_diff, 0.4*total_params)]
        # mean_abs_diff = torch.mean(mean_abs_diff)
        mean_abs_diff = -1
        percentile = [-1]

        # relative absolute difference
        rel_mean_abs_diff = ([torch.abs(param_1-param_2 - 10**-8).view(-1)/torch.abs(param_2 + 10**-8).view(-1) for param_1, param_2 in  zip(cur_model.parameters(), saved_model.parameters())])
        rel_mean_abs_diff = torch.cat(rel_mean_abs_diff, dim=0)
        # get percentile value 90th percentile 
        rel_percentile = [get_percentile(rel_mean_abs_diff, 0.9*total_params), get_percentile(rel_mean_abs_diff, 0.7*total_params), get_percentile(rel_mean_abs_diff, 0.4*total_params)]
        rel_mean_abs_diff = torch.mean(rel_mean_abs_diff)
    else:
        sim_loss = torch.tensor(0)
        rel_sim_loss = -1
    # final loss
    final_loss = float(wgt[2])*sim_loss + pred_loss

    return final_loss, pred_loss, sim_loss, rel_sim_loss, mean_abs_diff, percentile, rel_percentile, rel_mean_abs_diff, new_target


def train(train_iter, dev_iter, dev_adv_iter, model, args, static_model, eps, lp_norm='inf', model_eps_ball=False):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    
    # True when we are training on adv_data also
    # assuming we will only have dev_adv_iter when training on adv_data
    is_adv_train = True if len(dev_adv_iter.dataset) else False  
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target, adv_label = batch.text, batch.label, batch.adv_label
            feature.data = feature.data.t()
            # target.data  = target.data.sub_(1)  # batch first, index align
            
            #feature = feature.transpose(0,1)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            orig_pred_label = None
            if is_adv_train:
                assert static_model.training == False
                orig_pred_label = F.softmax(static_model(feature), dim=1)#.type(torch.FloatTensor)
                orig_pred_label = torch.argmax(orig_pred_label, dim=1)
                loss, pred_loss, sim_loss, rel_sim_loss, mean_abs_diff, percentile, rel_percentile, rel_mean_abs_diff, new_target = custom_loss_w_orig_label(logit, target, batch, args.weight, model, static_model, orig_pred_label)
            else:
                loss = F.cross_entropy(logit, target)
                pred_loss, sim_loss, rel_sim_loss, mean_abs_diff, percentile, rel_mean_abs_diff, rel_percentile = -1, -1, -1, -1, [-1], -1, [-1]
            
            loss.backward()
            optimizer.step()

            # project the weights to the Lp norm 
            if model_eps_ball:
                projection_lp_norm(model, static_model, eps, args, lp_norm)

            steps += 1
            if steps % args.log_interval == 0:
                if is_adv_train == True:
                    corrects = (torch.max(logit, 1)[1].view(new_target.size()).data == new_target.data).sum()
                else:
                    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

                accuracy = 100.0 * corrects/batch.batch_size
            
            if steps % args.test_interval == 0:
                print(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}, pred_loss: {:.5f}, sim_loss: {:.5f}, rel_sim_loss: {:.5f}, mean_abs_diff: {}, percentile: {}, rel_mean_diff: {}, rel_percentile: {}'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             pred_loss,
                                                                             sim_loss,
                                                                             rel_sim_loss,
                                                                             "{0:.2E}".format(mean_abs_diff),
                                                                             ["{0:.2E}".format(x) for x in percentile],
                                                                             "{0:.2E}".format(rel_mean_abs_diff),
                                                                             ["{0:.2E}".format(x) for x in rel_percentile]))

                dev_acc, dev_loss, _ = eval(dev_iter, model, static_model, args, is_adv_train=is_adv_train)
                
                if len(dev_adv_iter.dataset)!=0:
                    dev_adv_acc, dev_adv_loss, _ = eval(dev_adv_iter, model, static_model, args, is_adv_train=is_adv_train)
                else:
                    dev_adv_acc, dev_adv_loss = -1, -1
            
                print('\nNormal Dev - loss: {:.6f}  acc: {:.4f}'.format(dev_loss, dev_acc))
                print('Adversarial Dev - loss: {:.6f}  acc: {:.4f}\n'.format(dev_adv_loss, dev_adv_acc))

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best and not is_adv_train:
                        print(is_adv_train)
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0  and not is_adv_train:
                print(is_adv_train)
                save(model, args.save_dir, 'snapshot', steps)


def train_model_eps_ball(train_iter, dev_iter, dev_adv_iter, model, args, static_model, eps, lp_norm='inf', model_eps_ball=False):
    if args.cuda:
        model.cuda()

    if args.optim_algo.lower() == 'adam':
        print("adad")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_algo.lower() == 'sgd':
        print("sdsd")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = [-1,-1,-1] #dev_acc_rel, adv_dev_acc, dev_acc  
    best_adv_acc = [-1,-1,-1] #dev_acc_rel, adv_dev_acc, dev_acc
    last_step = 0
    model.train()
    
    # True when we are training on adv_data also
    # assuming we will only have dev_adv_iter when training on adv_data
    is_adv_train = True if len(dev_adv_iter.dataset) else False  
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            model.training = True
            static_model.training = False
            feature, target, adv_label = batch.text, batch.label, batch.adv_label
            feature.data = feature.data.t()
            # target.data  = target.data.sub_(1)  # batch first, index align
            
            #feature = feature.transpose(0,1)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            orig_pred_label = None
            if is_adv_train and args.train_on_base_model_label:
                assert static_model.training == False
                orig_pred_label = F.softmax(static_model(feature), dim=1)#.type(torch.FloatTensor)
                orig_pred_label = torch.argmax(orig_pred_label, dim=1)
                loss, new_target = custom_loss_I(logit, target, batch, args.weight, orig_pred_label)
            else:
                loss = F.cross_entropy(logit, target)
            
            loss.backward()
            optimizer.step()

            # project the weights to the Lp norm 
            if model_eps_ball:
                total_params = projection_lp_norm(model, static_model, eps, args, lp_norm)

            steps += 1
            if steps % args.log_interval == 0:
                if is_adv_train == True and args.train_on_base_model_label:
                    corrects = (torch.max(logit, 1)[1].view(new_target.size()).data == new_target.data).sum()
                else:
                    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

                accuracy = 100.0 * corrects/batch.batch_size
            
            if steps % args.test_interval == 0:
                if is_adv_train:
                    diff_tensor = torch.cat([(param_1-param_2).view(-1) for param_1, param_2 in  zip(model.parameters(), static_model.parameters())], dim=0)
                    x = torch.cat([param_2.view(-1) for param_2 in static_model.parameters()], dim=0)
                    
                    l2_norm_diff = torch.norm(diff_tensor)
                    l2_norm_orig = torch.norm(x)

                    linf_norm_diff = torch.norm(diff_tensor, float('inf'))
                    linf_norm_orig = torch.norm(x, float('inf'))

                    l1_norm_diff = torch.norm(diff_tensor,1)
                    l1_norm_orig = torch.norm(x,1)

                    # l2_norm_rel = torch.norm(diff_tensor)/torch.norm(x)
                    # linf_norm_rel = torch.norm(diff_tensor,float('inf'))/torch.norm(x, float('inf'))
                    # l1_norm_diff = torch.norm(diff_tensor, 1)
                    #rel_mean_abs_diff = ([torch.abs(param_1-param_2 - 10**-8).view(-1)/torch.abs(param_2 + 10**-8).view(-1) for param_1, param_2 in  zip(model.parameters(), static_model.parameters())])
                    #rel_mean_abs_diff = ([torch.abs(param_1-param_2 - 10**-8).view(-1)/torch.abs(param_2 + 10**-8).view(-1) for param_1, param_2 in  zip(model.parameters(), static_model.parameters())])
                    
                    # rel_mean_abs_diff = ([torch.abs(param_1-param_2).view(-1)/torch.abs(param_2 + torch.mean(x)).view(-1) for param_1, param_2 in  zip(model.parameters(), static_model.parameters())])
                    # rel_mean_abs_diff = torch.cat(rel_mean_abs_diff, dim=0)
                    # # get percentile value 90th percentile 
                    # rel_percentile = [get_percentile(rel_mean_abs_diff, 0.9*total_params), get_percentile(rel_mean_abs_diff, 0.7*total_params)]
                    # rel_mean_abs_diff = torch.mean(rel_mean_abs_diff)

                print(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}'.format(steps, loss.item(), accuracy))

                dev_acc, dev_loss, dev_acc_rel = eval(dev_iter, model, static_model, args, is_adv_train=is_adv_train)
                
                if len(dev_adv_iter.dataset)!=0:
                    dev_adv_acc, dev_adv_loss, adv_dev_acc_rel = eval(dev_adv_iter, model, static_model, args, is_adv_train=is_adv_train)
                    assert dev_adv_acc == adv_dev_acc_rel
                else:
                    dev_adv_acc, dev_adv_loss = -1, -1
            
                print('\nNormal Dev - loss: {:.6f}  acc: {:.4f}  {}, {}, {}, {}, {}, {}, {}\n'.format(dev_loss, dev_acc, dev_acc_rel, l1_norm_diff, l1_norm_orig, l2_norm_diff, l2_norm_orig, linf_norm_diff, linf_norm_orig))
                print('Adversarial Dev - loss: {:.6f}  acc: {:.4f}, {}, {}, {}, {}, {}, {}, {}\n'.format(dev_adv_loss, dev_adv_acc, dev_acc_rel, l1_norm_diff, l1_norm_orig, l2_norm_diff, l2_norm_orig, linf_norm_diff, linf_norm_orig))

                if dev_adv_acc > best_adv_acc[1]: # best accuracy according to adv_accuracy
                    best_adv_acc = [dev_acc_rel, dev_adv_acc, dev_acc, l1_norm_diff, l1_norm_orig, l2_norm_diff, l2_norm_orig, linf_norm_diff, linf_norm_orig]
                    # best_adv_acc[0] = dev_acc_rel
                    # best_adv_acc[1] = dev_adv_acc
                    # best_adv_acc[2] = dev_acc
                    # best_adv_acc[3] = rel_mean_abs_diff
                    # best_adv_acc += rel_percentile
                
                # best according to dev accuracy
                if dev_acc_rel > best_acc[0]: #still taking best on 
                    best_acc = [dev_acc_rel, dev_adv_acc, dev_acc, l1_norm_diff, l1_norm_orig, l2_norm_diff, l2_norm_orig, linf_norm_diff, linf_norm_orig]  
                    # best_acc[0] = dev_acc_rel
                    # best_acc[1] = dev_adv_acc
                    # best_acc[2] = dev_acc
                    # best_acc += rel_percentile

                    last_step = steps
                    if args.save_best and not is_adv_train:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0 and not is_adv_train:
                save(model, args.save_dir, 'snapshot', steps)
    
    return best_acc, best_adv_acc 


def eval(data_iter, model, static_model, args, is_adv_train=False):
    with torch.no_grad():
        # model.eval()
        # corrects, avg_loss = 0, 0
        # corrects_with_model_target = 0
        # for batch in data_iter:
        #     feature, target = batch.text, batch.label
        #     feature = feature.data.t()
        #     # target.data = target.data.sub_(1)
        #     if args.cuda:
        #         feature, target = feature.cuda(), target.cuda()

        #     logit = model(feature)
        #     if is_adv_train:
        #         # if is_adv_train, then update the target labels 
        #         # orig examples 
        #         orig_idx = (batch.adv_label == 0).type(torch.LongTensor)
        #         # adv examples 
        #         adv_idx  = (batch.adv_label == 1).type(torch.LongTensor)

        #         if IS_CUDA:
        #             orig_idx = orig_idx.type(torch.cuda.LongTensor)
        #             adv_idx = adv_idx.type(torch.cuda.LongTensor)

        #         orig_pred_label = F.softmax(static_model(feature), dim=1)#.type(torch.FloatTensor)
        #         orig_pred_label = torch.argmax(orig_pred_label, dim=1)
        #         new_target = orig_pred_label*orig_idx + target*adv_idx  

        #     # print(target)
        #     loss_with_orig_target  = F.cross_entropy(logit, target)
        #     if is_adv_train:
        #         loss_with_model_target = F.cross_entropy(logit, new_target)

        #     avg_loss += loss_with_orig_target.item()
        #     # print(torch.max(logit, 1)[1].view(target.size()).data[1:10])
        #     # print(target.data[1:10])
        #     corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        #     if is_adv_train:
        #         corrects_with_model_target += (torch.max(logit, 1)[1].view(new_target.size()).data == new_target.data).sum()
        # size = len(data_iter.dataset)
        # avg_loss /= size
        # accuracy = 100.0 * corrects/size
        # relative_accuracy = 100.0 * corrects_with_model_target/size

        return 0,0,0


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    # print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
