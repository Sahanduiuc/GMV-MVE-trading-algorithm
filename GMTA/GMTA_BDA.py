from TTool import *
import numpy as np
import pandas as pd

class GMTA_BDA:
    def __init__(
        self,
        scodes,
        period = 40,
        ws = None,
        method = 'max',
        quandl_apikey = None
        ):
        self.scodes = scodes
        assert period > 0
        assert method in ['max','min']
        self.period = int(period)
        self.method = method
        if ws is None:
            ws = [0.02 for i in range(len(scodes)-1)]
        assert len(ws) == len(scodes)-1
        self.ws = ws

    def one_trade(self,data):
        d = data.copy()
        mmap = {}
        for scode in self.scodes:
            x = pd.Series(np.zeros(len(self.scodes)),index = self.scodes)
            x[scode] = 1
            mmap[scode] = x
        mx = 0
        while len(d.columns) > 1:
            corr = None
            if self.method == 'max':
                corr = -d.corr() + 2*np.identity(len(d.columns))
            else:
                corr = d.corr()
            scode1 = corr.min().idxmin()
            scode2 = corr[scode1].idxmin()
            rho = d.corr()[scode1][scode2]

            d1 = d.pop(scode1)
            d2 = d.pop(scode2)
            s1 = d1.std(ddof = 0)
            s2 = d2.std(ddof = 0)
            r1 = d1.mean()
            r2 = d2.mean()

            a = s1**2 + s2**2 - 2*rho*s1*s2
            b = 2*rho*s1*s2 - 2*s2**2
            c = s2**2

            e = r1 - r2
            f = r2

            wgmv = min(max(-b/(2*a),0),1)
            wmve = min(max((b*f-2*c*e)/(b*e-2*a*f),0),1)

            w = self.ws[mx]*wgmv + (1-self.ws[mx])*wmve

            mmap[mx] = mmap[scode1]*w + mmap[scode2]*(1-w)
            d[mx] = d1*w + d2*(1-w)
            mx += 1
        return mmap[mx-1]



