"""
Author: Steve Paul 
Date: 3/9/22 """

class battery:

    def __init__(self,
                  battery_max = 100,
                  self_discharge_rate = 0.1):
        self.battery_max = battery_max
        self.current_battery_charge = battery_max
        self.self_discharge_rate = self_discharge_rate

    def update_charging(self, delta_t, charge_rate):
        self.current_battery_charge = max((1 - self.self_discharge_rate)*self.current_battery_charge + charge_rate*delta_t, self.battery_max)

    def update_discharging(self, delta_t, discharge_rate):
        self.current_battery_charge  = min((1 - self.self_discharge_rate)*self.current_battery_charge - discharge_rate*delta_t, 0)

    def batter_reset(self):
        self.current_battery_charge = self.battery_max

class eVTOL:

    def __init__(self,
                 id,
                 location = None,
                 speed = 40, #miles/hr
                 max_passenger = 6,
                 current_passengers=0,
                 charging = False,
                 landing_time = 0.25,
                 take_off_time = 0.25,
                 next_flight_time = 6.00,
                 next_decision_time = 6.00,
                 status = 0
                 ):
        self.id = id
        self.current_location = location
        self.next_location = None
        self.speed = speed
        self.max_passenger = max_passenger
        self.battery = battery()
        self.current_passengers = current_passengers
        self.charging = charging
        self.status = status # 0 -charging. 1- idle, 2 - engaged
        self.next_flight_time = next_flight_time
        self.next_decision_time = next_decision_time
        self.landing_time = landing_time
        self.take_off_time = take_off_time


    def update_location(self, new_location):
        self.current_location = new_location

    def update_current_passengers(self, n_passengers):
        self.current_passengers = n_passengers

    def update_battery_charge(self, delta_t, charging_rate):
        self.battery.update_charging(delta_t, charging_rate)
    def update_battery_discharge(self, delta_t, discarge_rate):
        self.battery.update_charging(delta_t, discarge_rate)

    def update_status(self, new_status):
        self.status = new_status