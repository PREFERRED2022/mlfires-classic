from csv import DictWriter
import os

def writemetrics(metrics, mean_metrics, hpresfile, allresfile):
    if not os.path.exists(os.path.dirname(hpresfile)):
        os.makedirs(os.path.dirname(hpresfile))
    writeheader = True if not os.path.isfile(hpresfile) else False
    with open(hpresfile, 'a') as _f:
        dw = DictWriter(_f, fieldnames=mean_metrics.keys(), quotechar='"')
        if writeheader:
            dw.writeheader()
        dw.writerow(mean_metrics)
    writeheader = True if not os.path.isfile(allresfile) else False
    with open(allresfile, 'a') as _f:
        dw = DictWriter(_f, fieldnames=metrics[0].keys(), quotechar='"')
        if writeheader:
            dw.writeheader()
        for m in metrics:
            dw.writerow(m)
