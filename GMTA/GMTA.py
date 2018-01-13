use_quandl = True
try:
    import quandl
except:
    use_quandl = False
use_portfoliomgr = True
try:
    from Portfolio import Portfolio
except:
    use_portfoliomgr = False
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class GMTA:
    def __init__(
        self,
        scodes,
        Rf = 0.01/310,
        period = 150,
        no_short = True,
        optimizer_tol = 1e-15,
        optimizer_maxiter = 600,
        quandl_apikey = None,
    ):
        global use_quandl,use_portfoliomgr
        if not use_quandl:
            assert quandl_apikey is None
        self.scodes = scodes
        self.Rf = Rf
        self.period = period
        self.no_short = no_short
        self.tol = optimizer_tol
        self.maxiter = optimizer_maxiter
        self.data = pd.DataFrame(columns = scodes)
        self.gmv = None
        self.mve = None
        self.apikey = quandl_apikey
            
    def update(self,data=None):
        if data is not None:
            self.data = data
        self.cov = self.data.cov()
        self.std = self.data.std(ddof = 0)
        self.mean = self.data.mean()
        
    def global_variance(self,w):
        assert len(w) == len(self.scodes)
        res = np.dot(np.matmul(w,self.cov.values),w)
        return np.sqrt(res)
    
    def sharpe_ratio(self,w):
        assert len(w) == len(self.scodes)
        return (np.dot(w,self.std.values)-self.Rf)/self.global_variance(w)
    
    def GMV(self,no_short = None,tol = None,maxiter = None):
        if no_short is None:
            no_short = self.no_short
        if tol is None:
            tol = self.tol
        if maxiter is None:
            maxiter = self.maxiter
        w0 = np.array([1.0/len(self.scodes)]*len(self.scodes))
        params = {
            'fun':lambda x:self.global_variance(x),
            'x0':w0,
            'method':'SLSQP',
            'constraints':{
                'type':'eq',
                'fun':(lambda x:sum(x)-1)
            },
            'tol':tol,
            'options':{
                'maxiter':maxiter
            }
        }
        if no_short:
            params['bounds'] = [(0,1) for i in range(len(self.scodes))]
        gmv = minimize(**params)
        if gmv.message != 'Optimization terminated successfully.':
            print("Error during GMV minimize:")
            print(gmv.message)
        self.Wgmv = gmv.x
        self.Rgmv = np.dot(gmv.x,self.mean)
        self.Vgmv = gmv.fun
        self.gmv = gmv
        
    def MVE(self,no_short = None,tol = None,maxiter = None):
        if no_short is None:
            no_short = self.no_short
        if tol is None:
            tol = self.tol
        if maxiter is None:
            maxiter = self.maxiter
        w0 = np.array([1.0/len(self.scodes)]*len(self.scodes))
        params = {
            'fun':lambda x:-self.sharpe_ratio(x),
            'x0':w0,
            'method':'SLSQP',
            'constraints':{
                'type':'eq',
                'fun':(lambda x:sum(x)-1)
            },
            'tol':tol,
            'options':{
                'maxiter':maxiter
            }
        }
        if no_short:
            params['bounds'] = [(0,1) for i in range(len(self.scodes))]
        mve = minimize(**params)
        if mve.message != 'Optimization terminated successfully.':
            print("Error during MVE minimize:")
            print(mve.message)
        self.Wmve = mve.x
        self.Rmve = np.dot(mve.x,self.mean)
        self.Vmve = self.global_variance(mve.x)
        self.mve = mve
        
    def GMc(self):
        wmve = self.Wmve.reshape(-1,1)
        wgmv = self.Wgmv.reshape(-1,1)
        U = np.matmul(wgmv,wmve.T)
        return np.dot(U.flatten(),self.cov.values.flatten())
    
    def risk_return(self,w=0.02):
        cov = self.GMc()
        r = w*self.Rgmv + (1-w)*self.Rmve
        d = np.sqrt(w**2*self.Vgmv**2 + (1-w)**2*self.Vmve**2 + 2*w*(1-w)*cov)
        return d,r

    def one_trade(self,data,w=0.02):
        data = data[self.scodes]
        self.data = data.iloc[-self.period:]
        self.update()
        self.GMV()
        self.MVE()
        return self.Wgmv * w + self.Wmve * (1-w)
        
    def trading_simulator(self,data,w=0.02):
        data = data[self.scodes]
        ws = [np.zeros(len(self.scodes))]
        rs = []
        assert len(data) > self.period
        for i in range(len(data)-self.period):
            self.data = data.iloc[i:i+self.period]
            self.update()
            self.GMV()
            self.MVE()
            rs.append(np.dot(ws[-1],self.data.iloc[-1].values))
            ws.append(w*self.Wgmv + (1-w)*self.Wmve)
        return rs,ws
    
    def quandl_test_data_generator(self):
        global use_quandl
        if self.apikey is None:
            print('quandl is not avaliable or no apikey provided, cannot use this function')
            return
        quandl.ApiConfig.api_key = self.apikey
        res = pd.DataFrame()
        for scode in self.scodes:
            cl = quandl.get("EOD/"+scode.replace(".","_"))['Close']
            res[scode] = (cl-cl.shift(1))/cl.shift(1)
        quandl.ApiConfig.api_key = None
        return res.dropna()
    
    def quandl_today_data_generator(self):
        global use_quandl
        if self.apikey is None:
            print('quandl is not avaliable or no apikey provided, cannot use this function')
            return
        res = pd.DataFrame()
        quandl.ApiConfig.api_key = self.apikey
        for scode in self.scodes:
            cl = quandl.get("EOD/"+scode.replace(".","_"),rows = self.period+1)["Close"]
            res[scode] = (cl-cl.shift(1))/cl.shift(1)
        quandl.ApiConfig.api_key = None
        return res.dropna()
    
    def robinhood_one_trade_per_day_with_quandl(self,p):
        assert isinstance(p,Portfolio)
        data = self.quandl_today_data_generator()
        p.portfolio_record_lock.acquire()
        idxs = p.portfolio_record.index
        p.portfolio_record_lock.release()
        for x in idxs:
            assert x in self.scodes
        w_target = self.one_trade(data)
        w_current = pd.Series(p.get_weights(*self.scodes)).loc[self.scodes].values
        w_diff = w_target - w_current
        s_diff = pd.Series(
            ((w_diff*p.get_market_value())//p.quote_last_price(*self.scodes)).astype(int),
            index = self.scodes
        )
        for scode in self.scodes:
            n = s_diff.loc[scode]
            if n>0:
                p.market_buy(scode,int(n))
            elif n<0:
                p.market_sell(scode,int(-n))
        
            