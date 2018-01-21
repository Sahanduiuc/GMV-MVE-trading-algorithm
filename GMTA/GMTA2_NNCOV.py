from .GMTA import GMTA
import pandas as pd
import numpy as np
class GMTA2_NNCOV(GMTA):
    def __init__(
        self,
        scodes,
        Rf = 0.01/310,
        period = 30,
        no_short = True,
        optimizer_tol = 1e-15,
        optimizer_maxiter = 600,
        quandl_apikey = None,
        period2 = 30,
        w2 = 0.02,
        w3 = 0.02
        ):
        params = locals()
        self.period2 = params.pop('period2')
        self.w2 = params.pop('w2')
        self.w3 = params.pop('w3')
        GMTA.__init__(**params)

    def one_trade(self,datax = None):
        if datax is None:
            datax = self.data
        data = datax[:self.period]
        data = data[self.scodes]
        ss = []
        d0 = data
        rs = []
        while len(d0):
            d1 = pd.DataFrame()
            while (d0.cov()<0).sum().sum():
                scode = (d0.cov()<0).sum().argmax()
                d1[scode] = d0.pop(scode)
            rs.append(d0)
            ss.append(list(d0.columns))
            d0 = d1
        aas = pd.DataFrame()
        ws = {}
        for s,dt in zip(ss,rs):
            g = GMTA(
                scodes = s,
                Rf = self.Rf,
                period = self.period2,
                no_short = self.no_short,
                optimizer_tol = self.tol,
                optimizer_maxiter = self.maxiter,
                quandl_apikey = self.apikey
                )
            a,b,_,_ = g.trading_simulator(datax[s],self.w2)
            aas[','.join(s)] = a
            ws[','.join(s)] = pd.Series(b[-1],index = s)
        gt = GMTA(scodes = list(aas.columns),period = self.period2)
        W = pd.Series(gt.one_trade(aas,self.w3),index = aas.columns)
        res = pd.Series(np.zeros(len(self.scodes)),index = self.scodes)
        for s in W.index:
            res.loc[s.split(',')] = W.loc[s]*ws[s]
        return res

    def trading_simulator(self,data):
        s_period = self.period+self.period2
        L = len(data)
        ws = [np.zeros(len(self.scodes))]
        rs = []
        for i in range(L-s_period-1):
            datax = data.iloc[i:i+s_period]
            w = self.one_trade(datax)
            rs.append(np.dot(ws[-1],datax.iloc[-1]))
            ws.append(w)
        return rs,ws,[],[]



