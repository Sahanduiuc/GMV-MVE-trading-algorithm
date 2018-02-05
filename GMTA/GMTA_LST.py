from GMTA import GMTA_BDA
from GMTA import GMTA
from TTool import *

class GMTA_LST(GMTA):
    def __init__(
        self,
        scodes,
        s_period = 20,
        s_ws = None,
        l_period = 160,
        l_ws = None,
        s_method = 'max',
        l_method = 'max',
        no_short = True,
        quandl_apikey = None,
        w = None
    ):
        GMTA.__init__(self,scodes,0,l_period,True,1e-15,600,quandl_apikey)
        assert s_period > 0
        assert l_period > s_period
        if w is None:
            w = 0.5
        self.w = w
        self.scodes = scodes
        self.s_g = GMTA_BDA(scodes,s_period,s_ws,s_method,no_short,quandl_apikey)
        self.l_g = GMTA_BDA(scodes,l_period,l_ws,l_method,no_short,quandl_apikey)
    
    def one_trade(self,data = None):
        assert data is not None
        assert len(data)>=self.period
        data = data[self.scodes]

        w_s = self.s_g.one_trade(data)
        w_l = self.l_g.one_trade(data)
        
        return self.w*w_s + (1-self.w)*w_l
        
    def one_suggestion_qd_rh(self,pmgr,pname):
        p = pmgr.portfolios[pname]
        data = self.quandl_today_data_generator()['data']
        p.portfolio_record_lock.acquire()
        idxs = p.portfolio_record.index
        p.portfolio_record_lock.release()
        for x in idxs:
            assert x in self.scodes
        w_target = self.one_trade(data = data)
        s_target = pd.Series(
            ((w_target*p.get_market_value())/p.quote_last_price(*self.scodes)).astype(int),
            index = self.scodes
        )

        s_diff = (s_target - p.portfolio_record['SHARES'].loc[s_target.index].fillna(0)).astype(int)

        return s_diff        
        
    def trading_simulator(self,data):
        data = data[self.scodes]
        ws = [pd.Series(np.zeros(len(self.scodes)),index = self.scodes)]
        rs = []
        assert len(data) > self.period
        for i in range(len(data)-self.period):
            d = data.iloc[i:i+self.period]
            wres = self.one_trade(d)
            rs.append(np.dot(ws[-1],d.iloc[-1].values))
            ws.append(wres)
        m = np.array(rs)+1
        for i in range(1,len(m)):
            m[i] *= m[i-1]
        return rs,ws,m,[]