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
        no_short = True,
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
        self.apikey = quandl_apikey

    def one_trade(self,data):
        d = data.copy()[self.scodes]
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
            wgmv = -b/(2*a)
            wmve = (b*f-2*c*e)/(b*e-2*a*f)
            if no_short:
                wgmv = min(max(wgmv,0),1)
                wmve = min(max(wmve,0),1)


            w = self.ws[mx]*wgmv + (1-self.ws[mx])*wmve

            mmap[mx] = mmap[scode1]*w + mmap[scode2]*(1-w)
            d[mx] = d1*w + d2*(1-w)
            mx += 1
        return mmap[mx-1]


    def quandl_today_data_generator(self):
        """
        generate todays data with quandl
        """
        global use_quandl
        if self.apikey is None:
            print('quandl is not avaliable or no apikey provided, cannot use this function')
            return
        res = pd.DataFrame()
        res_cp = pd.DataFrame()
        quandl.ApiConfig.api_key = self.apikey
        for scode in self.scodes:
            dt = quandl.get("EOD/"+scode.replace(".","_"),rows = self.period+1)
            cl = dt['Close']
            split = dt['Split']
            sidx = split[split!=1].index
            for idx in sidx:
                cl.loc[:idx] = cl.loc[:idx]/split.loc[idx]
                cl.loc[idx] = cl.loc[idx]*split.loc[idx]
            res[scode] = (cl-cl.shift(1))/cl.shift(1)
            res_cp[scode] = cl
        quandl.ApiConfig.api_key = None
        res = res.dropna()
        res_cp = res_cp.loc[res.index]
        return {'data':res,'data_p':res_cp}

    def trading_simulator(self,data):
        data = data[self.scodes]
        ws = [pd.Series(np.zeros(len(self.scodes)),index = self.scodes)]
        rs = []
        assert len(data) > self.period
        for i in range(len(data)-self.period):
            d = data.iloc[i:i+self.period]
            wres = self.one_trade(d)
            rs.append(np.dot(wres,d.iloc[-1].values))
            ws.append(wres)
        m = np.array(rs)+1
        for i in range(1,len(m)):
            m[i] *= m[i-1]
        return rs,ws,m,[]

    def one_suggestion_qd_rh(self,pmgr,pname):
        p = pmgr.portfolios[pname]
        data = self.quandl_today_data_generator()['data']
        p.portfolio_record_lock.acquire()
        idxs = p.portfolio_record.index
        p.portfolio_record_lock.release()
        for x in idxs:
            assert x in self.scodes
        w_target = self.one_trade(data = data)
        w_current = pd.Series(p.get_weights(*self.scodes)).loc[self.scodes].values
        w_diff = w_target - w_current
        s_diff = pd.Series(
            ((w_diff*p.get_market_value())/p.quote_last_price(*self.scodes)).astype(int),
            index = self.scodes
        )
        return s_diff

    def real_simulator(self,data,datap,inifund = 25000):

        ps_s = [pd.Series(np.zeros(len(data.columns)),index = data.columns)]
        mv_s = [inifund]
        bp_s = []
        bp = inifund

        for i in range(len(data) - self.period-2):
            d = data.iloc[i:self.period + i ]
            p = datap.iloc[self.period + i]
            w = self.one_trade(d)
            mv = bp + pd.np.dot(ps_s[-1],p)

            ps = (w*mv*0.95/p).astype(int)

            ps_diff = ps - ps_s[-1]


            for scode in ps_diff.index:
                if ps_diff[scode] > 0:
                    bp -= abs(ps_diff[scode]) * p[scode] * 1.005
                if ps_diff[scode] < 0:
                    bp += abs(ps_diff[scode]) * p[scode] * 0.995
            mv = bp + pd.np.dot(ps,p)
            ps_s.append(ps)
            mv_s.append(mv)
            bp_s.append(bp)
        return ps_s,mv_s,bp_s

