use_quandl = True
try:
    import quandl
except:
    use_quandl = False
use_portfoliomgr = True
try:
    from Robinhood import Robinhood
    from Portfolio import PortfolioMgr
    from Portfolio import Portfolio
    from TTool import sell_first_decision_exe
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
        period = 30,
        no_short = True,
        optimizer_tol = 1e-15,
        optimizer_maxiter = 600,
        quandl_apikey = None,
    ):
        """
        class for trading algorithm

        scodes (list): list of stock symbols to be considered by algorithm, 
            the portfolio to apply this algorithm can not own stocks not in this list but not verse vice
        Rf (float): risk free rate, daily or minutely base on yourt algorithm frequent
        period (int): the last `period` entries will be used as history
        no_short (bool): wheather shorting strategy is allowed, robinhood account dot allow storting stocks
        optimizer_tol (float): the tolerance for optimizer to seek GMV and MVE
        optimizer_maxiter (int): maximum iterations allowed for optimizer to find solution
        quandl_apikey (string|None): optional, put None here if your data wont come from quandl 

        """
        global use_quandl,use_portfoliomgr
        if not use_quandl:
            assert quandl_apikey is None
        for scode in scodes:
            while scodes.count(scode) > 1:
                scodes.remove(scode)
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
        self.raw_data = pd.DataFrame()

    def update(self,data = None):
        """
        update data and statistics info of data

        data (DataFrame|None): data of historical price change in each stock 
        """
        if data is not None:
            self.data = data[self.scodes]
        self.cov = self.data.cov()
        self.std = self.data.std(ddof = 0)
        self.mean = self.data.mean()
        
    def global_variance(self,w):
        """
        calculate global variance of portfolio if assets weight is w
        
        w (array): array of weights of assets

                GV = sqrt( w COV w^T )

        """
        assert len(w) == len(self.scodes)
        res = np.dot(np.matmul(w,self.cov.values),w)
        return np.sqrt(res)
    
    def sharpe_ratio(self,w):
        """
        calculate sharpe ratio of portifolio if assets weight is w

        w (array): array of weights of assets

        """
        assert len(w) == len(self.scodes)
        return (np.dot(w,self.mean.values)-self.Rf)/self.global_variance(w)
    
    def GMV(self,no_short = None,tol = None,maxiter = None):
        """
        calculate global min variance weight of assets

        no_short (bool|None): allow short or not
        tol (float|None): tolerance for optimizer
        maxiter (int|None): maxiter for optimizer

        W_{GMV} = argmin_{w}(sqrt(w COV w^T))
        """
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
        """
        calculate weight of assets with maximum sharp ratio

        W_{MVE} = argmax_{w}(w^T std - Rf)
        """
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
        """
        calculate covariance between mve and gmv
        
        GMc = < W_{GMV} W_{MVE}^T , COV) 
        """
        wmve = self.Wmve.reshape(-1,1)
        wgmv = self.Wgmv.reshape(-1,1)
        U = np.matmul(wgmv,wmve.T)
        return np.dot(U.flatten(),self.cov.values.flatten())
    
    def risk_return(self,w = 0.02):
        """
        calculate risk and return for a weight
        """
        cov = self.GMc()
        r = w*self.Rgmv + (1-w)*self.Rmve
        d = np.sqrt(w**2*self.Vgmv**2 + (1-w)**2*self.Vmve**2 + 2*w*(1-w)*cov)
        return d,r

    def one_trade(self,data = None,w = 0.02):
        """
        calculate weight for assets
        """
        if data is not None:
            data = data[self.scodes]
        self.data = data.iloc[-self.period:]
        self.update()
        self.GMV()
        self.MVE()
        return pd.Series(self.Wgmv * w + self.Wmve * (1-w),index = self.scodes)
        
    def trading_simulator(self,data,w = 0.02):
        """
        simulate trading process with time series data
        """
        data = data[self.scodes]
        ws = [np.zeros(len(self.scodes))]
        rs = []
        v = []
        assert len(data) > self.period
        for i in range(len(data)-self.period):
            self.data = data.iloc[i:i+self.period]
            self.update()
            self.GMV()
            self.MVE()
            rs.append(np.dot(ws[-1],self.data.iloc[-1].values))
            ws.append(w*self.Wgmv + (1-w)*self.Wmve)
            _,v_ = self.risk_return()
            v.append(v_)
        m = np.array(rs)+1
        for i in range(1,len(m)):
            m[i] *= m[i-1]
        return rs,ws,m,v

    def mixed_trading_simulator(self,data = None,idata = None,robin_un = None,robin_pd = None):
        if data is None:
            data = self.quandl_today_data_generator()['data_p']
        if idata is None:
            idata = self.rh_intraday_data_generator(robin_un,robin_pd,span = 'day')['data_p']
        qdata = (idata - idata.shift(1))/idata.shift(1)
        qdata.iloc[0] = (idata.iloc[0] - data.iloc[-1])/data.iloc[-1]
        idata = (idata-data.iloc[-1])/data.iloc[-1]
        data = ((data-data.shift(1))/data.shift(1)).dropna()
        now = Portfolio.get_time()

        ws = [np.zeros(len(self.scodes))]
        rs = []
        v = []

        
        for i in range(len(idata)):
            data.loc[now] = idata.iloc[i]
            w = self.one_trade(data)
            rs.append(np.dot(ws[-1],qdata.iloc[i]))
            ws.append(w)


        m = np.array(rs)+1
        for i in range(1,len(m)):
            m[i] *= m[i-1]
        return rs,ws,m,v




    
    def quandl_test_data_generator(self,trade_on = 'close',extent = False):
        """
        generate test data with quandl
        """
        assert trade_on in ['close','open']
        global use_quandl
        if self.apikey is None:
            print('quandl is not avaliable or no apikey provided, cannot use this function')
            return
        quandl.ApiConfig.api_key = self.apikey
        res = pd.DataFrame()
        res_cp = pd.DataFrame()
        for scode in self.scodes:
            try:
                dt = quandl.get("EOD/"+scode.replace(".","_"))
            except Exception as e:
                print('error {}'.format(scode))
                print(e)
                self.scodes.remove(scode)
            cl = dt['Adj_Open']
            if trade_on == 'close':
            	cl = dt['Adj_Close']

            res[scode] = (cl-cl.shift(1))/cl.shift(1)
            res_cp[scode] = cl
        if extent:
        	res = res.fillna(0)
        else:
        	res = res.dropna()
        res_cp = res_cp.loc[res.index]
        quandl.ApiConfig.api_key = None
        return {'data':res,'data_p':res_cp}
    
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
            cl = dt['Adj_Close']

            res[scode] = (cl-cl.shift(1))/cl.shift(1)
            res_cp[scode] = cl
        quandl.ApiConfig.api_key = None
        res = res.dropna()
        res_cp = res_cp.loc[res.index]
        return {'data':res,'data_p':res_cp}
    
    def rh_intraday_data_generator(self,robin_un,robin_pd,interval = '5minute',span = 'week'):
        trader = Robinhood()
        trader.login(robin_un,robin_pd)
        dfs = trader.get_historical_quotes(self.scodes,"5minute","week","regular")
        data = pd.DataFrame()
        data_cp = pd.DataFrame()
        sd = zip(self.scodes,dfs)
        for scode,df in sd:
            cl = df["Close"]
            c = (cl-cl.shift(1))/cl.shift(1)
            data[scode] = c
            data_cp[scode] = cl
        data = data.dropna()
        data_cp = data_cp.loc[data.index]
        return {'data':data,'data_p':data_cp}


    def one_suggestion_qd_rh(self,pmgr,pname,w=0.02):
        p = pmgr.portfolios[pname]
        data = self.quandl_today_data_generator()['data']
        p.portfolio_record_lock.acquire()
        idxs = p.portfolio_record.index
        p.portfolio_record_lock.release()
        for x in idxs:
            assert x in self.scodes
        w_target = self.one_trade(data = data,w = w)
        s_target = pd.Series(
            ((w_target*p.get_market_value())/p.quote_last_price(*self.scodes)).astype(int),
            index = self.scodes
        )
        s_diff = (s_target - p.portfolio_record['SHARES'].loc[s_target.index].fillna(0)).astype(int)
        return s_diff


    def market_decision_exe(self,p,s_diff):
        for scode in self.scodes:
            n = s_diff.loc[scode]
            if n>0:
                p.market_buy(scode,int(n))
            elif n<0:
                p.market_sell(scode,int(-n))

    def algo_header(self,**args):
        assert isinstance(args['pmgr'],PortfolioMgr)
        assert args['pname'] in args['pmgr'].portfolios
        if not args['args']["call_from_mgr"]:
            args['pmgr'].schedule(
                algo = self,
                method = args['method'],
                portfolio_name = args['pname'],
                freq = args['freq'],
                misc = args['misc']
            )
            return False
        return True

    def mixed_trade_with_quandl_and_robinhood(
        self,
        pmgr,
        pname,
        args = {
            "call_from_mgr" : False,
        },misc = {
            "w" : 0.02,
            "cancel_count" : 5
        }
    ):
        params = locals()
        params.pop('self')
        params['method'] = 'mixed_trade_with_quandl_and_robinhood'
        params['freq'] = 2
        if not self.algo_header(**params):
            return
        w = misc['w']
        p = pmgr.portfolios[pname]
        data_d = self.quandl_today_data_generator()
        data = data_d['data']
        data_p = data['data_p']
        data_n = p.quote_last_price(*self.scodes)
        dd = data_p[self.scode].iloc[-1]
        d = pd.DataFrame([(data_n - dd)/dd],columns = self.scodes)
        data = data.append(d)
        p.portfolio_record_lock.acquire()
        idxs = p.portfolio_record.index
        p.portfolio_record_lock.release()
        for x in idxs:
            assert x in self.scodes
        w_target = self.one_trade(data = data,w = w)
        w_current = pd.Series(p.get_weights(*self.scodes)).loc[self.scodes].values
        w_diff = w_target - w_current
        s_diff = pd.Series(
            ((w_diff*p.get_market_value())/p.quote_last_price(*self.scodes)).astype(int),
            index = self.scodes
        )
        p.log_lock.acquire()
        p.log.append("{}: decision made : {}".format(Portfolio.get_time(),str(s_diff)))
        p.log_lock.release()

        sell_first_decision_exe(p,s_diff)


    def one_trade_per_day_with_quandl_and_robinhood(
        self,
        pmgr,
        pname,
        args = {
            "call_from_mgr" : False
        },misc = {
            "w" : 0.02,
            "cancel_count" : 5
        }
    ):
        """
        apply this algorithm on a robinhood portfolio once a day
        pmgr (PortfolioMgr): portfolio manager
        pname (str): name of a portfolio
        args (dict): information for portfolio mgr to schedule this algorithm
        misc (dict): parameters for this method only
        """
        #TODO remove duplicated
        params = locals()
        params.pop('self')
        params['method'] = 'one_trade_per_day_with_quandl_and_robinhood'
        params['freq'] = 1440
        if not self.algo_header(**params):
            return

        w = misc['w']
        p = pmgr.portfolios[pname]
        data = self.quandl_today_data_generator()['data']
        p.portfolio_record_lock.acquire()
        idxs = p.portfolio_record.index
        p.portfolio_record_lock.release()
        for x in idxs:
            assert x in self.scodes
        w_target = self.one_trade(data = data,w = w)
        w_current = pd.Series(p.get_weights(*self.scodes)).loc[self.scodes].values
        w_diff = w_target - w_current
        s_diff = pd.Series(
            ((w_diff*p.get_market_value())/p.quote_last_price(*self.scodes)).astype(int),
            index = self.scodes
        )
        p.log_lock.acquire()
        p.log.append("{}: decision made : {}".format(Portfolio.get_time(),str(s_diff)))
        p.log_lock.release()

        sell_first_decision_exe(p,s_diff)

    


    def intraday_trading_with_robinhood(
        self,
        pmgr,
        pname,
        args = {
            "frequent" : 2,
            "call_from_mgr" : False
        },
        misc = {
            "w" : 0.02,
            "cancel_count" : 15
        }
    ):
        """
        frequently trading with this algorithm, 
        your should have over 20000$ in your account for this method otherwise robinhood will keeps you from intraday trading
        the frequent parameter is measured in minutes

        """
        
        params = locals()
        #print(params)
        params.pop('self')
        params['method'] = 'one_trade_per_day_with_quandl_and_robinhood'
        params['freq'] = args['frequent']

        if not self.algo_header(**params):
            return

        w = misc['w']
        p = pmgr.portfolios[pname]
        self.raw_data.loc[Portfolio.get_time()] = p.quote_last_price(*self.scodes)
        if len(self.raw_data)<self.period:
            return
        DDX = ((self.row_data - self.raw_data.shift(1))/self.raw_data.shift(1)).iloc[-self.period:]
        
        w_target = self.one_trade(data = DDX,w = w)
        w_current = pd.Series(p.get_weights(*self.scodes)).loc[self.scodes].values
        w_diff = w_target - w_current
        s_diff = pd.Series(
            ((w_diff*p.get_market_value())/self.data.loc[Portfolio.get_time()].values).astype(int),
            index = self.scodes
        )
        p.log_lock.acquire()
        p.log.append("{}: decision made : {}".format(Portfolio.get_time(),str(s_diff)))
        p.log_lock.release()

        sell_first_decision_exe(p,s_diff)





