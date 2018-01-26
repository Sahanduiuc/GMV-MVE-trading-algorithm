from TTool import *
import numpy as np
import pandas as pd
from .GMTA import GMTA

class GMTA_WTD(GMTA):
    def __init__(
        self,
        scodes,
        Rf = 0.01/310,
        period = 30,
        no_short = True,
        optimizer_tol = 1e-15,
        optimizer_maxiter = 600,
        quandl_apikey = None,
        ws = None
        ):
        params = locals()
        self.ws = params.pop('ws')
        GMTA.__init__(**params)
        if self.ws is None:
            self.ws = pd.Series(np.ones(self.period))
    def update(self,data = None):
        if data is not None:
            data = data[self.scodes]
            self.data = data
        self.mean = self.data.apply(lambda x:np.average(x,weights = self.ws))
        self.std = ((self.data - self.mean)**2).apply(lambda x:np.sqrt(np.average(x,weights=self.ws)))
        self.cov = pd.DataFrame(np.cov(self.data.T,aweights=self.ws),index = self.scodes,columns = self.scodes)
