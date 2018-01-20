from .GMTA import GMTA
class GMTA2_NNCOV(GMTA):
    def __init__(
        self,
        scodes,
        Rf = 0.01/310,
        period = 30,
        no_short = True,
        optimizer_tol = 1e-15,
        optimizer_maxiter = 600,
        quandl_apikey = None
        ):
        params = locals()
        GMTA.__init__(**params)

    def update(self,data = None):
        if data is None:
            data = self.data
        data = data[self.scodes]
        ss = []
        d0 = data
        rs = []
        while len(cs):
            d1 = pd.DataFrame()
            while (d0.cov()<0).sum().sum():
                scode = (d0.cov()<0).sum().argmax()
                d1[scode] = d0.pop(scode)
            rs.append(d0)
            ss.append(d0.columns)
            d0 = d1
        return ss,rs



