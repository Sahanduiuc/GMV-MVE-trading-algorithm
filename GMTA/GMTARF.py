from .GMTA import GMTA

class GMTARF(GMTA):
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
        param = locals()
        GMTA.__init__(**param)
    def quandl_test_data_generator(self):
        try:
            self.scode.remove("RISKFREE")
        except:
            pass
        data_d = super(GMTARF,self).quandl_test_data_generator()
        data_d["data"]["RISKFREE"] = 0
        data_d["data_p"]["RISKFREE"] = 1
        self.scodes.append("RISKFREE")
        return data_d