"""
Author: Steve Paul 
Date: 3/9/22 """

class Vertiport:

    def __init__(self,
                 id,
                 location,
                 max_evotls_park = 5,
                 max_evtol_charge = 5,
                 n_evtol_parked = 0,
                 n_evtols_charging = 0
                 ):
        self.id = id
        self.location = location
        self.max_evotls_park = max_evotls_park
        self.max_evtol_charge = max_evtol_charge
        self.n_evtol_parked = n_evtol_parked
        self.n_evtols_charging = n_evtols_charging

    def update_parked_evtols(self, n_evtols):
        self.n_evtol_parked = self.n_evtol_parked + n_evtols
        if self.n_evtol_parked < 0 or self.n_evtol_parked > self.max_evotls_park:
            print("Error")
            raise ValueError


    def update_charging_evtols(self, n_evtols):
        self.n_evtols_charging = self.n_evtols_charging + n_evtols
        if self.n_evtols_charging < 0 or self.n_evtols_charging > self.max_evtol_charge:
            print("Error")
            raise ValueError

