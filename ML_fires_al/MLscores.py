from sklearn.metrics import confusion_matrix

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
