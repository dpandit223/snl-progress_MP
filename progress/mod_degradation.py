import numpy as np
import rainflow
import pybamm

class BESS_Degradation:

    def __init__(self, cell_chemistry):
        # Example stress factor coefficients and references
        self.cell_chemistry = cell_chemistry
        self.static_params = {}
        self.static_params["SOC_ref"] =  0.5
        self.static_params["T_ref"] =  25
        self.static_params["DoD_ref"] =  1
        if cell_chemistry == "LMO":
            self.static_params["kD1"] = 1.4e5
            self.static_params["kD2"] = -0.501e-1
            self.static_params["kD3"] = -1.23e5
            self.static_params["ks"] = 1.04
            self.static_params["C_ref"] = 1.0
            self.static_params["kC"] = 0.1
            self.static_params["kT"] = 6.93e-2
            self.static_params["alpha_SEI"] =  0.0572
            self.static_params["beta_SEI"] = 121
            
        if cell_chemistry == "LFP":
            self.static_params["alpha_SEI"] =  0.001687
            self.static_params["beta_SEI"] = 189.68
            self.static_params["kD1"] = 3.1335e-5
            self.static_params["kD2"] = 0.4678
            self.static_params["kT1"] = -0.1126
            self.static_params["kT2"] = 0.5593
            self.static_params["kT3"] = -0.0605
            self.static_params["kC1"] = -1.051
            self.static_params["C_ref"] = 0.5

        if cell_chemistry == "NMC":
            self.static_params["alpha_SEI"] =  6.63728501e-02
            self.static_params["beta_SEI"] = 7.99233420e+02
            self.static_params["kD1"] = 4.09744125e-04
            self.static_params["kD2"] = 4.83030040e+00
            self.static_params["kD3"] = -1.67720426e-06
            self.static_params["kD4"] = 4.84632785e-01
            self.static_params["kT1"] = -0.0645748
            self.static_params["kC1"] = 0
            self.static_params["C_ref"] = 0.5 #doesnt matter
            
        if cell_chemistry == "NCA":
            self.static_params["alpha_SEI"] =  5.27276491e-02
            self.static_params["beta_SEI"] = 3.49791602e+02
            self.static_params["kD1"] = 3.61680962e-04
            self.static_params["kD2"] = 3.05101133
            self.static_params["kT1"] = 0
            self.static_params["kC1"] = -1.69336521
            self.static_params["kC2"] = 0.00191186
            self.static_params["kC3"] = 0.91492666
            self.static_params["C_ref"] = 1 
    
    @staticmethod
    def evaluate_C_rates(charge_profile, discharge_profile, S_limit, ess_eff):
        charge_rate = np.zeros(len(charge_profile))
        discharge_rate = np.zeros(len(charge_profile))
        for i in range(len(charge_profile)):
            charge_rate[i] = -1*charge_profile[i]*ess_eff/S_limit
            discharge_rate[i] = discharge_profile[i]/S_limit
        overall_rates = charge_rate + discharge_rate*-1
        return discharge_rate, overall_rates

    @staticmethod
    def evaluate_cell_temp(c_rate_profile, init_soc = 0.5):

        model = pybamm.lithium_ion.SPMe(options={
            "thermal": "lumped",
            "SEI": "none",
            "lithium plating": "none",
            "loss of active material": "none"
        })

        # 2. Load and modify parameter values
        params = pybamm.ParameterValues("OKane2022") #Identical cell to the SNL NMC cells
        
        steps = []
        for current_crate in c_rate_profile:
            if current_crate > 1e-5:
                steps.append(f"Charge at {current_crate}C for 60 minutes")
            elif current_crate < -1e-5:
                steps.append(f"Discharge at {abs(current_crate)}C for 60 minutes")
            else:
                steps.append(f"Rest for 60 minutes")

        experiment = pybamm.Experiment(steps)
        output_var = ["Volume-averaged cell temperature [K]"]
        # 4. Create and solve the simulation
        solver_baseline = pybamm.IDAKLUSolver(rtol=1e-2, atol=1e-2)
        sim = pybamm.Simulation(model, experiment=experiment, output_variables=output_var, parameter_values=params, solver=solver_baseline)
        sim.solve(initial_soc=init_soc)

        # 5. Extract and plot temperature
        T = sim.solution["X-averaged cell temperature [K]"]
        time = sim.solution["Time [s]"]

        hours = np.arange(1, len(c_rate_profile)+1)  # 1,2,...,24 hours
        seconds = hours * 3600
        indices = [np.argmin(np.abs(time.entries - t)) for t in seconds]
        return T.entries[indices] - 273.15    

    def update_instance(self, soc_profile, c_rates, temp):

        self.soc_profile = soc_profile
        self.C_rates = c_rates
        self.T = temp
    
    def fDoD(self, DoD, cyc_cumsum):
        """Depth of Discharge (DoD) stress factor."""
        if self.cell_chemistry == "LMO":
            return (self.static_params["kD1"] * (DoD+1e-20) **self.static_params["kD2"] + self.static_params["kD3"]) ** -1 
        if self.cell_chemistry == "LFP" or self.cell_chemistry == "NCA":
            return self.static_params["kD1"]*np.exp(self.static_params["kD2"]*(DoD-self.static_params["DoD_ref"]))
        if self.cell_chemistry == "NMC":
            return (self.static_params["kD1"]*np.exp(self.static_params["kD2"]*(DoD-self.static_params["DoD_ref"]))
                    + (self.static_params["kD3"]*(cyc_cumsum)**self.static_params["kD4"]) * (DoD - self.static_params["DoD_ref"]))

    def fs(self, SOC):
        """State of Charge (SOC) stress factor. LMO only"""
        if self.cell_chemistry == "LMO":
            return np.exp(self.static_params["ks"] * (SOC - self.static_params["SOC_ref"]))

    def fC(self, C, cyc_cumsum):
        """C-rate stress factor."""
        if self.cell_chemistry == "LFP" or self.cell_chemistry == "NMC":
            return np.exp(self.static_params["kC1"] * (C - self.static_params["C_ref"]))
        if self.cell_chemistry == "NCA":
            return (np.exp(self.static_params["kC1"] * (C - self.static_params["C_ref"]))+
                          self.static_params["kC2"]*(cyc_cumsum)**self.static_params["kC3"]* (C - self.static_params["C_ref"]))
        
    def fT(self, T, cyc_cumsum):
        """Temperature stress factor."""
        if self.cell_chemistry == "LMO": 
            return np.exp(self.static_params["kT"] * (T - self.static_params["T_ref"]) * (self.static_params["T_ref"] / T ))
        if self.cell_chemistry == "LFP":
            return np.exp(self.static_params["kT1"]*(T - self.static_params["T_ref"]) * (self.static_params["T_ref"] / T))  \
                + (self.static_params["kT2"]*(cyc_cumsum)**self.static_params["kT3"])*(T-self.static_params["T_ref"])*self.static_params["T_ref"]/T
        if self.cell_chemistry == "NMC" or self.cell_chemistry == "NCA":
            return np.exp(self.static_params["kT1"] * (T - self.static_params["T_ref"]) * (self.static_params["T_ref"] / T))
        
    def calculate_cycle_data(self):
        
        dod = []
        av_SoC = []
        num_cycles = []
        cycle_temp = []
        C_cyc = []
        updated_C_rates = self.C_rates.copy()
        
        for cyc_ampl, mean_val, cyc_no, cyc_start_time, cyc_end_time in rainflow.extract_cycles(self.soc_profile): 
            dod.append(cyc_ampl)
            av_SoC.append(mean_val)
            num_cycles.append(cyc_no)
            cycle_temp.append(np.mean(self.T[cyc_start_time:cyc_end_time]))
            C_cyc.append(np.max(self.C_rates[cyc_start_time:cyc_end_time]))

        for n, c_rate in enumerate(C_cyc):
            if c_rate == 0:
                C_cyc[n] = self.static_params["C_ref"] #so that for charing and regulation has no effect on discharge C-rate stres factor

        return (np.array(dod),np.array(av_SoC),np.array(num_cycles),np.array(cycle_temp),np.array(C_cyc))

    def calculate_cycle_degradation(self):
        
        DoD,av_SoC,num_cycles,cycle_temp,C_cyc = self.calculate_cycle_data()
        self.num_cycles = num_cycles #To calculate EFC later on
        cumulative_cycles = np.cumsum(num_cycles)
        fDoD_values = self.fDoD(DoD,cumulative_cycles)
        fs_values = self.fs(av_SoC)
        fT_values = self.fT(cycle_temp,cumulative_cycles)
        fC_values = self.fC(C_cyc,cumulative_cycles)
        if self.cell_chemistry == "LMO":
            degradation = np.sum(fDoD_values * fs_values * fT_values * num_cycles)
        else:
            degradation = np.sum(fDoD_values * fT_values * fC_values * num_cycles)
        return degradation

    def calculate_total_degradation(self):
        
        fd = self.calculate_cycle_degradation()
        self.L = 1 - self.static_params["alpha_SEI"] * np.exp(-self.static_params["beta_SEI"] * fd) - (1 - self.static_params["alpha_SEI"]) * np.exp(-fd)
        
# # Example usage (placeholder values)
# if __name__ == "__main__":
    
#     # Example cycle life inputs
#     SoC_profile = np.array([0.5,0.55,0.6,0.7,0.9,0.9,0.9,0.9,0.7,0.5,0.3,0.1,0.3,0.5,0.5])
#     T = np.ones(len(SoC_profile))*25#np.array([25, 27, 30, 25, 25, 25, 25, 28 ,30,25,25,25])
    
#     deg_model = BESS_Degradation("LFP")
#     deg_model.update(SoC_profile,T)
#     deg_model.calculate_total_degradation()
#     print("Total degradation (L):", deg_model.L)
    