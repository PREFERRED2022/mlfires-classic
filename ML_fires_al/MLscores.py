from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from tensorflow.keras.metrics import AUC

def recall(tp,fn):
    if tp+fn == 0:
        return -1000
    return tp/(tp+fn)

def precision(tp,fp):
    if tp+fp == 0:
        return -1000
    return tp/(tp+fp)

def accuracy(tp,tn,fp,fn):
    if tp+tn+fp+fn == 0:
        return -1000
    return (tp+tn)/(tp+tn+fp+fn)

def f1(tp,fp,fn):
    if recall(tp,fn)+precision(tp,fp) == 0:
        return -1000
    return 2*recall(tp,fn)*precision(tp,fp)/(recall(tp,fn)+precision(tp,fp))

def npv(tn,fn):
    if tn+fn == 0:
        return -1000
    return tn/(tn+fn)

def hybridrecall(w1, w0, rec1, rec0, hybtype = 'hybrid'):
    if hybtype == 'hybrid':
        if rec1 > 0 and rec0 > 0:
            return (w1 + w0) / (w1 / rec1 + w0 / rec0)
        else:
            return -1000
    elif hybtype == 'NH':
        return w1*rec1 + w0*rec0
    else:
        return -1000

def calc_all_hybrids(rec_1, rec_0, debug=True):
    hybrid1 = hybridrecall(2, 1, rec_1, rec_0)
    hybrid2 = hybridrecall(5, 1, rec_1, rec_0)
    nh1 = hybridrecall(2, 1, rec_1, rec_0, 'NH')
    nh2 = hybridrecall(5, 1, rec_1, rec_0, 'NH')
    if debug:
        print("hybrid 1 : %.2f" % hybrid1)
        print("hybrid 2 : %.2f" % hybrid2)
        print("NH 1 : %.2f" % nh1)
        print("NH 2 : %.2f" % nh2)
    return hybrid1, hybrid2, nh1, nh2

def cmvals(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0]==1 and y_true[0]==0:
        tn = cm[0,0]
        fp = 0
        fn = 0
        tp = 0
    elif cm.shape[0]==1 and y_true[0]==1:
        tn = 0
        fp = 0
        fn = 0
        tp = cm[0,0]
    elif cm.shape[0]==2:
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
    else:
        tn=fp=fn=tp=None
    return tn, fp, fn, tp

def calc_metrics(y, y_scores, y_pred, numaucthres=200, debug=True):
    if debug:
        print("calulating merics from scores (sklearn)")
        print("calulating tn, fp, fn, tp")
    tn, fp, fn, tp = cmvals(y, y_pred)
    if debug:
        print("tn : %d, fp : %d, fn : %d, tp : %d" % (tn, fp, fn, tp))
    if debug:
        print("calulating auc...")
    if numaucthres>0:
        aucmetric = AUC(num_thresholds=numaucthres)
        aucmetric.update_state(y, y_scores[:, 1])
        auc = float(aucmetric.result())
    else:
        auc = 0.0
    if debug:
        print("auc : %.2f" % auc)
    if debug:
        print("calulating accuracy...")
    acc = accuracy_score(y, y_pred)
    #acc_0 = accuracy_score(1 - y, 1 - y_pred)
    if debug:
        print("accuracy : %.2f" % acc)
        #print("accuracy 0 : %.2f" % acc_0)
    if debug:
        print("calulating recall...")
    rec_1 = recall_score(y, y_pred)
    rec_0 = recall_score(1 - y, 1 - y_pred)
    if debug:
        print("recall 1 : %.2f" % rec_1)
        print("recall 0 : %.2f" % rec_0)
    if debug:
        print("calulating precision...")
    prec_1 = precision_score(y, y_pred)
    prec_0 = precision_score(1 - y, 1 - y_pred)
    if debug:
        print("precision 1 : %.2f" % prec_1)
        print("precision 0 : %.2f" % prec_0)
    if debug:
        print("calulating f1 score...")
    f1_1 = f1_score(y, y_pred)
    f1_0 = f1_score(1 - y, 1 - y_pred)
    if debug:
        print("f1 1 : %.2f" % f1_1)
        print("f1 0 : %.2f" % f1_0)
    if debug:
        print("calulating hybrids...")
    hybrid1, hybrid2, nh1, nh2 = calc_all_hybrids(rec_1, rec_0, debug)
    # tp0 = tn1 tn0 = tp1 fp0 = fn1 fn0 = fp1
    return auc, acc, prec_1, prec_0, rec_1, rec_0, f1_1, f1_0, hybrid1, hybrid2, nh1, nh2, tn, fp, fn, tp

def metrics_dict(auc, acc, prec_1, prec_0, rec_1, rec_0, f1_1, f1_0, hybrid1, hybrid2, nh1, nh2, tn, fp, fn, tp, metricset):
    dictmetrics = { 'accuracy %s' % metricset: acc,
        'precision 1 %s' % metricset: prec_1,
        'recall 1 %s' % metricset: rec_1,
        'f1-score 1 %s' % metricset: f1_1,
        'precision 0 %s' % metricset: prec_0,
        'recall 0 %s' % metricset: rec_0,
        'f1-score 0 %s' % metricset: f1_0,
        'auc %s' % metricset: auc,
        'hybrid1 %s'%metricset: hybrid1,
        'hybrid2 %s'%metricset: hybrid2,
        'NH 1 %s' % metricset: nh1,
        'NH 2 %s' % metricset: nh2,
        'TN %s' % metricset: tn,
        'FP %s' % metricset: fp,
        'FN %s' % metricset: fn,
        'TP %s' % metricset: tp,
    }
    return dictmetrics


def calc_metrics_custom(tn, fp, fn, tp, y_scores, y, numaucthres=200, debug=True):
    if debug:
        print("calulating merics (custom)")
    if debug:
        print("(input) tn : %d, fp : %d, fn : %d, tp : %d" % (tn, fp, fn, tp))
    if debug:
        print("calulating auc...")
    if numaucthres > 0:
        aucmetric = AUC(num_thresholds=numaucthres)
        aucmetric.update_state(y, y_scores[:, 1])
        auc = float(aucmetric.result())
    else:
        auc = 0.0
    if debug:
        print("auc : %.2f" % auc)
    ##############################################
    # tp0 = tn1, tn0 = tp1, fp0 = fn1, fn0 = fp1 #
    ##############################################
    if debug:
        print("calulating accuracy...")
    acc = accuracy(tp, tn, fp, fn)
    #acc_0 = accuracy(tn, tp, fn, fp)
    if debug:
        print("accuracy : %.2f" % acc)
        #print("accuracy 0 : %.2f" % acc_0)
    if debug:
        print("calulating recall ...")
    rec_1 = recall(tp, fn)
    rec_0 = recall(tn, fp)
    if debug:
        print("recall 1 : %.2f" % rec_1)
        print("recall 0 : %.2f" % rec_0)
    if debug:
        print("calulating precision...")
    prec_1 = precision(tp, fp)
    prec_0 = precision(tn, fn)
    if debug:
        print("precision 1 : %.2f" % prec_1)
        print("precision 0 : %.2f" % prec_0)
    if debug:
        print("calulating f1_score...")
    f1_1 = f1(tp, fp, fn)
    f1_0 = f1(tn, fn, fp)
    if debug:
        print("f1 1 : %.2f" % f1_1)
        print("f1 0 : %.2f" % f1_0)
    if debug:
        print("calulating hybrids ...")
    hybrid1, hybrid2, nh1, nh2 = calc_all_hybrids(rec_1, rec_0, debug)

    return auc, acc, prec_1, prec_0, rec_1, rec_0, f1_1, f1_0, hybrid1, hybrid2, nh1, nh2, tn, fp, fn, tp

def metrics_aggr(metrics, mean_metrics):
    for m in metrics[0]:
        if isinstance(metrics[0][m], str):
            continue
        metricsum = sum([item.get(m, 0) for item in metrics if item.get(m) >= 0])
        cmvalsts = ['TN', 'FP', 'FN', 'TP']
        if any([st in m for st in cmvalsts]):
            mean_metrics[m] = metricsum
        else:
            mean_metrics[m] = metricsum / len(metrics)
    return mean_metrics
