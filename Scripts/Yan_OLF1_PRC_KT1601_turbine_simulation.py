# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:19:41 2025

@author: sukumar
"""
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy.optimize import minimize_scalar
import ipywidgets as widgets
from IPython.display import display

file_path = "..\\Data"


#%% Compressor k1601 flow and power calculation
df_K1601 = pd.read_excel(file_path +"/"+"Yan_OLF1_PRC_K1601_compressor_data.xlsx")
colnames = df_K1601.iloc[0,:]
df_K1601.columns = colnames

df_K1601 = df_K1601.iloc[1:].reset_index(drop=True)
df_K1601.set_index("DateTime",inplace =True, drop=True)
df_K1601 = df_K1601.apply(pd.to_numeric, errors='coerce')
df_K1601.dropna(how="any",inplace=True)

# Get density of propylene (C3H6) at T and P
Density_1st_stage =[]
rho_1st_stage_flow = []
for i in range(len(df_K1601)):
    T_K = df_K1601["K1601 1ST STAGE TEMP"].iloc[i] + 273.15
    P_Pa = df_K1601["K1601 1ST STAGE PRESSURE"].iloc[i] * 1000 + 1e5
    rho_1st_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Propylene')
    volumetric_flow = df_K1601["K1601 1ST STAGE FLOW"].iloc[i] * 1000 / rho_1st_stage
    rho_1st_stage_flow.append(volumetric_flow)
    Density_1st_stage.append(rho_1st_stage)
       
df_K1601["Density_1st_stage"] = Density_1st_stage
df_K1601["K1601 VOL FLOW 1ST STAGE"] = rho_1st_stage_flow

Density_2nd_stage =[]
rho_2nd_stage_flow = []
for i in range(len(df_K1601)):
    T_K = df_K1601["K1601 2ND STAGE TEMP"].iloc[i] + 273.15
    P_Pa = df_K1601["K1601 2ND STAGE PRESSURE"].iloc[i]* 1000 + 1e5
    rho_2nd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Propylene')
    volumetric_flow = (df_K1601["K1601 2ND STAGE FLOW"].iloc[i] + df_K1601["K1601 2ND STAGE ADDITIONAL FLOW"].iloc[i]) * 1000 / rho_2nd_stage
    rho_2nd_stage_flow.append(volumetric_flow)
    Density_2nd_stage.append(rho_2nd_stage)
    
df_K1601["Density_2nd_stage"] = Density_2nd_stage   
df_K1601["K1601 VOL FLOW 2ND STAGE"] = rho_2nd_stage_flow

Density_3rd_stage =[]
rho_3rd_stage_flow = []

for i in range(len(df_K1601)):
    T_K = df_K1601["K1601 3RD STAGE TEMP"].iloc[i] + 273.15
    P_Pa = df_K1601["K1601 3RD STAGE PRESSURE"].iloc[i] * 1000 + 1e5
    rho_3rd_stage = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Propylene')
    volumetric_flow = df_K1601["K1601 3RD STAGE FLOW"].iloc[i] * 1000 / rho_3rd_stage
    rho_3rd_stage_flow.append(volumetric_flow)
    Density_3rd_stage.append(rho_3rd_stage)

df_K1601["Density_3rd_stage"] = Density_3rd_stage
df_K1601["K1601 VOL FLOW 3RD STAGE"] = rho_3rd_stage_flow

file_path_K1601 = "..\\Results"
df_K1601.to_excel(file_path_K1601 +"/"+ "Yan_OLF1_PRC K1601_compressor_simulated_data.xlsx")

#Compressor power calculation based on isentropic enthalpy at outlet pressure, keeping entropy constant (i.e., isentropic compression).
eta = 0.60  # Assume Compressor efficiency
K1601_1st_stage_comp_estimated_power =[]
# Step 2: Get inlet enthalpy and entropy
for i in range(len(df_K1601)):
    h1 = PropsSI('H', 'T',df_K1601["K1601 1ST STAGE TEMP"].iloc[i] + 273.15 , 'P', df_K1601["K1601 1ST STAGE PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg
    s1 = PropsSI('S', 'T',df_K1601["K1601 1ST STAGE TEMP"].iloc[i] + 273.15 , 'P', df_K1601["K1601 1ST STAGE PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg.K
    #Get isentropic outlet enthalpy (h2s at P2, s1)
    T2s = PropsSI('T', 'P', df_K1601["K1601 2ND STAGE PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
    h2s = PropsSI('H', 'P', df_K1601["K1601 2ND STAGE PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
    h2 = h1 + (h2s - h1) / eta
    power = ((df_K1601["K1601 1ST STAGE FLOW"].iloc[i] * 1000)/3600) * (h2 - h1)/1e6  # KW
    K1601_1st_stage_comp_estimated_power.append(power)
    
df_K1601["K1601_1st_stage_comp_estimated_power_MW"] = K1601_1st_stage_comp_estimated_power

# 2nd stage compressor power
K1601_2nd_stage_comp_estimated_power =[]
# Step 2: Get inlet enthalpy and entropy
for i in range(len(df_K1601)):
    h1 = PropsSI('H', 'T',df_K1601["K1601 2ND STAGE TEMP"].iloc[i] + 273.15 , 'P', df_K1601["K1601 2ND STAGE PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg
    s1 = PropsSI('S', 'T',df_K1601["K1601 2ND STAGE TEMP"].iloc[i] + 273.15 , 'P', df_K1601["K1601 2ND STAGE PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg.K
    #Get isentropic outlet enthalpy (h2s at P2, s1)
    T2s = PropsSI('T', 'P', df_K1601["K1601 3RD STAGE PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
    h2s = PropsSI('H', 'P', df_K1601["K1601 3RD STAGE PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
    h2 = h1 + (h2s - h1) / eta
    power = (((df_K1601["K1601 2ND STAGE FLOW"].iloc[i] + df_K1601["K1601 2ND STAGE ADDITIONAL FLOW"].iloc[i]) * 1000)/3600) * (h2 - h1)/1e6  # KW
    K1601_2nd_stage_comp_estimated_power.append(power)
    
df_K1601["K1601_2nd_stage_comp_estimated_power_MW"] = K1601_2nd_stage_comp_estimated_power

# 3rd stage compressor power
K1601_3rd_stage_comp_estimated_power =[]
# Step 2: Get inlet enthalpy and entropy
for i in range(len(df_K1601)):
    h1 = PropsSI('H', 'T',df_K1601["K1601 3RD STAGE TEMP"].iloc[i] + 273.15 , 'P', df_K1601["K1601 3RD STAGE PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg
    s1 = PropsSI('S', 'T',df_K1601["K1601 3RD STAGE TEMP"].iloc[i] + 273.15 , 'P', df_K1601["K1601 3RD STAGE PRESSURE"].iloc[i] * 1000+ 1e5, 'Propylene')  # J/kg.K
    #Get isentropic outlet enthalpy (h2s at P2, s1)
    T2s = PropsSI('T', 'P', df_K1601["K1601 3RD STAGE FINAL PRESSURE"].iloc[i]* 1000 + 1e5, 'S', s1, 'Propylene')  # isentropic outlet temp
    h2s = PropsSI('H', 'P', df_K1601["K1601 3RD STAGE FINAL PRESSURE"].iloc[i]* 1000+ 1e5, 'S', s1, 'Propylene')  # J/kg
    h2 = h1 + (h2s - h1) / eta
    power = ((df_K1601["K1601 3RD STAGE FLOW"].iloc[i] * 1000)/3600) * (h2 - h1)/1e6  # KW
    K1601_3rd_stage_comp_estimated_power.append(power)
    
df_K1601["K1601_3rd_stage_comp_estimated_power_MW"] = K1601_3rd_stage_comp_estimated_power

df_K1601["K1601_Total_estimated_power_MW"] = df_K1601["K1601_1st_stage_comp_estimated_power_MW"]+df_K1601["K1601_2nd_stage_comp_estimated_power_MW"]+df_K1601["K1601_3rd_stage_comp_estimated_power_MW"]
df_K1601 = df_K1601.round(2)

file_path_K1601 = "..\\Results"
df_K1601.to_excel(file_path_K1601 +"/"+ "Yan_OLF1_PRC K1601_compressor_simulated_data.xlsx",freeze_panes=(1,0))

#%% KT-1601 dynamic simulation

df_KT1601 = pd.read_excel(file_path + "/" + "Yan_OLF1_PRC_KT1601_turbine_data.xlsx")
colnames = df_KT1601.iloc[0,:]
df_KT1601.columns = colnames
df_KT1601 = df_KT1601.iloc[1:].reset_index(drop=True)
df_KT1601.set_index("DateTime",inplace =True, drop=True)
df_KT1601 = df_KT1601.apply(pd.to_numeric, errors='coerce')

steam_enthalpy_KJ_Kg = []
steam_entropy_KJ_KgK =[]
outlet_ethalpy_KJ_Kg =[]
outlet_isentropic_ethalpy_KJ_Kg =[]
power_gen_extraction_MW = []
power_gen_exhaust_MW =[]
Turbine_power_MW_based_on_EE = []
Turbine_power_MW_based_on_steam_flow = []
Specific_steam_consumption_MT_MW =[]
#turbine_efficiency =[]

for i in range(len(df_KT1601)):
    Steam_flow_TPH = df_KT1601['KT-1601 steam flow'].iloc[i]
    Condensate_flow_TPH = df_KT1601['KT-1601 condensate flow'].iloc[i]
    Extraction_flow_TPH = df_KT1601['KT-1601 Extraction flow'].iloc[i]
    
    Steam_flow_Kg_hr       = Steam_flow_TPH *1000
    Extraction_flow_Kg_hr  = Extraction_flow_TPH *1000
    Condensate_flow_Kg_hr  = Condensate_flow_TPH *1000

    # Steam Inlet conditions
    P_steam = df_KT1601['Steam pressure'].iloc[i]*1000       # Pa
    T_steam = df_KT1601['Steam Temp'].iloc[i] +273.15   # K
    T_sat_steam = PropsSI('T', 'P', P_steam, 'Q', 1, 'Water') - 273.15 # degC

    # Extraction pressure
    Pe = df_KT1601['Extraction Pressure'].iloc[i]*1000          # Pa 
    T_sat_extraction = PropsSI('T', 'P', Pe, 'Q', 1, 'Water') # K

    # Condenser pressure
    Pc = df_KT1601['Condensate Pressure'].iloc[i]*1000
    Tc_actual = df_KT1601['Condensate Temperature'].iloc[i] +273.15  # K (actual exhaust steam temp)
    T_sat_condensate = PropsSI('T', 'P', Pc, 'Q', 0, 'Water') # degC
    
    # Steam (inlet) Enthalpy calculation 
    h_steam = PropsSI('H', 'P', P_steam, 'T', T_steam,'Water')/1000   # kJ/kg
    s_steam = PropsSI('S', 'P', P_steam, 'T', T_steam, 'Water')/1000  # KJ/kg·K
    
    # Actual outlet enthalpies
    # Actual outlet enthalpies
    # he = PropsSI('H', 'P', Pe, 'T', Te_actual, 'Water')/1000  # Extracted steam (actual)
    he = PropsSI('H', 'P', Pe, 'T', T_sat_extraction + 0.01, 'Water')/1000  # Extracted steam (actual)
    #hc = PropsSI('H', 'P', Pc, 'T', Tc_actual, 'Water')/1000  # Condenser outlet (actual)
    hc_liquid = PropsSI('H', 'P', Pc, 'T', T_sat_condensate - 0.01, 'Water')/1000  # Condenser outlet (actual)
    dryness_fraction = 0.92
    hc_vapor = PropsSI('H', 'P', Pc, 'T', T_sat_condensate + 0.01, 'Water')/1000  # Condenser outlet (actual)

    hc = hc_liquid + hc_vapor*dryness_fraction

    Net_heat_release = Steam_flow_Kg_hr*h_steam -Extraction_flow_Kg_hr*he - Condensate_flow_Kg_hr*hc
    power_gen_extraction = Extraction_flow_Kg_hr*(h_steam-he)/3600/1000
    power_gen_exhaust = Condensate_flow_Kg_hr*(h_steam-hc)/3600/1000
    Turbine_power_MW_EE = power_gen_extraction + power_gen_exhaust
    Specific_steam_consumption = Steam_flow_TPH/(Turbine_power_MW_EE) # MT/MW

    def isentropic_enthalpy(P_target, s_in):
        # Keep everything in J/kg·K
        s_f = PropsSI("S", "P", P_target, "Q", 0, "Water")/1000
        s_g = PropsSI("S", "P", P_target, "Q", 1, "Water")/1000
    
        if s_f < s_in < s_g:
            x = (s_in - s_f) / (s_g - s_f)
            h_f = PropsSI("H", "P", P_target, "Q", 0, "Water")
            h_g = PropsSI("H", "P", P_target, "Q", 1, "Water")
            h_iso = h_f + x * (h_g - h_f)
        else:
            h_iso = PropsSI("H", "P", P_target, "S", s_in, "Water")
    
        return h_iso / 1000  # Convert to kJ/kg


    # Isentropic outlet enthalpies
    he_s = isentropic_enthalpy(Pe, s_steam)
    hc_s = isentropic_enthalpy(Pc, s_steam)
    
    # Mass-weighted outlet enthalpies
    h2_actual = (Extraction_flow_Kg_hr * he + Condensate_flow_Kg_hr * hc) / Steam_flow_Kg_hr
    h2s_ideal = (Extraction_flow_Kg_hr  * he_s + Condensate_flow_Kg_hr * hc_s) / Steam_flow_Kg_hr
    
    # Turbine isentropic efficiency
    efficiency = (h_steam - h2_actual) / (h_steam - h2s_ideal)*100
    
    turbine_power_MW_SF = (h_steam - h2_actual) * (Steam_flow_Kg_hr / (3600*1000))  # Convert kg/hr to kg/s
    
    steam_enthalpy_KJ_Kg.append(h_steam)
    steam_entropy_KJ_KgK.append(s_steam)
    outlet_ethalpy_KJ_Kg.append(h2_actual)
    outlet_isentropic_ethalpy_KJ_Kg.append(h2s_ideal)
    power_gen_extraction_MW.append(power_gen_extraction)
    power_gen_exhaust_MW.append(power_gen_exhaust)
    Turbine_power_MW_based_on_EE.append(Turbine_power_MW_EE)
    Turbine_power_MW_based_on_steam_flow.append(turbine_power_MW_SF)
    Specific_steam_consumption_MT_MW.append(Specific_steam_consumption)
    #turbine_efficiency.append(efficiency)
    
df_KT1601["steam_enthalpy_KJ_Kg"] = steam_enthalpy_KJ_Kg
df_KT1601["steam_entropy_KJ_KgK"] = steam_entropy_KJ_KgK
df_KT1601["outlet_ethalpy_KJ_Kg"] = outlet_ethalpy_KJ_Kg
df_KT1601["outlet_isentropic_ethalpy_KJ_Kg"] = outlet_isentropic_ethalpy_KJ_Kg
df_KT1601["power_gen_extraction_MW"] = power_gen_extraction_MW
df_KT1601["power_gen_exhaust_MW"] = power_gen_exhaust_MW
df_KT1601["Turbine_power_MW_based_on_EE"] = Turbine_power_MW_based_on_EE
df_KT1601["Turbine_power_MW_based_on_steam_flow"] = Turbine_power_MW_based_on_steam_flow
df_KT1601["Specific_steam_consumption_MT_MW"] = Specific_steam_consumption_MT_MW
#df_KT1601["turbine_efficiency"] = turbine_efficiency
df_KT1601 =df_KT1601.round(2)

file_path_KT1601 = "..\\Results"
df_KT1601.to_excel(file_path_KT1601 +"/"+ "Yan_OLF1_PRC KT1601_turbine_simulated_data.xlsx",freeze_panes=(1,0))

#%% optimization of steam and extraction flow

df_Turbine_condenser_merge = df_K1601.merge(df_KT1601, left_on =df_K1601.index,right_on = df_KT1601.index)
df_Turbine_condenser_merge.set_index("key_0",inplace=True)
df_Turbine_condenser_merge.index.name = "DateTime"

from scipy.optimize import minimize_scalar

optimized_extraction = []
calculated_steam_flow = []
matched_power_EE = []
matched_power_SF = []
matched_h2_actual = []

def match_actual_power(row):
    try:
        # Steam inlet enthalpy
        P_steam = row["Steam pressure"] * 1000  # Pa
        T_steam = row["Steam Temp"] + 273.15    # K
        h_steam = PropsSI("H", "P", P_steam, "T", T_steam, "Water") / 1000  # kJ/kg
     
        # Actual extracted steam enthalpy (he)
        Pe = row["Extraction Pressure"] * 1000  # Pa
        # Te_actual = row["Extraction Temperature"] + 273.15  # K
        T_sat_extraction = PropsSI('T', 'P', Pe, 'Q', 1, 'Water') # K
        he = PropsSI('H', 'P', Pe, 'T', T_sat_extraction + 0.01, 'Water')/1000  # Extracted steam (actual)
        
        # Actual condenser steam enthalpy (hc)
        Pc = row["Condensate Pressure"] * 1000  # Pa
        # Tc_actual = row["Condensate Temperature"] + 273.15  # K
        T_sat_condensate = PropsSI('T', 'P', Pc, 'Q', 0, 'Water') # degC
        hc_liquid = PropsSI('H', 'P', Pc, 'T', T_sat_condensate - 0.01, 'Water')/1000  # Condenser outlet (actual)
        dryness_fraction = 0.92
        hc_vapor = PropsSI('H', 'P', Pc, 'T', T_sat_condensate + 0.01, 'Water')/1000  # Condenser outlet (actual)
    
        hc = hc_liquid + hc_vapor*dryness_fraction
    
        condensate_flow_TPH = row["KT-1601 condensate flow"]
        condensate_flow_Kg_hr = condensate_flow_TPH * 1000
        actual_power = row["K1601_Total_estimated_power_MW"]
    
        def objective(extraction_flow_TPH):
            extraction_flow_Kg_hr = extraction_flow_TPH * 1000
            steam_flow_Kg_hr = extraction_flow_Kg_hr + condensate_flow_Kg_hr
    
            power = ((extraction_flow_Kg_hr * (h_steam - he)) + 
                     (condensate_flow_Kg_hr * (h_steam - hc))) / 3600 / 1000
            return (power - actual_power) ** 2
    
        result = minimize_scalar(objective, bounds=(10, 350), method='bounded')
    
        if result.success:
            ef_opt = result.x
            ef_Kg_hr = ef_opt * 1000
            sf_opt = ef_opt + condensate_flow_TPH
            sf_Kg_hr = sf_opt * 1000
            # Matched power via energy balance
            power_matched_EE = ((ef_Kg_hr * (h_steam - he)) + (condensate_flow_Kg_hr * (h_steam - hc))) / 3600 / 1000

            # Mass-weighted outlet enthalpy
            h2_actual = (ef_Kg_hr * he + condensate_flow_Kg_hr * hc) / sf_Kg_hr

            # Power via inlet/outlet enthalpy and total flow
            power_matched_SF = (h_steam - h2_actual) * (sf_Kg_hr / 3600 / 1000)

            return ef_opt, sf_opt, power_matched_EE, power_matched_SF, h2_actual
        else:
            return (row["KT-1601 Extraction flow"], row["KT-1601 steam flow"],
                    row["Turbine_power_MW_based_on_EE"], row["Turbine_power_MW_based_on_steam_flow"],
                    None)
    except Exception as e:
        print(f"Error at index {row.name}: {e}")
        return (row["KT-1601 Extraction flow"], row["KT-1601 steam flow"],
                row["Turbine_power_MW_based_on_EE"], row["Turbine_power_MW_based_on_steam_flow"],
                None)
    
# Loop and apply
for _, row in df_Turbine_condenser_merge.iterrows():
    ef, sf, power_EE, power_SF, h2 = match_actual_power(row)
    optimized_extraction.append(ef)
    calculated_steam_flow.append(sf)
    matched_power_EE.append(power_EE)
    matched_power_SF.append(power_SF)
    matched_h2_actual.append(h2)

# Save to DataFrame
df_Turbine_condenser_merge["Optimized_Extraction_flow_TPH"] = optimized_extraction
df_Turbine_condenser_merge["Calculated_Steam_flow_TPH"] = calculated_steam_flow
df_Turbine_condenser_merge["Matched_Turbine_power_MW_EE"] = matched_power_EE
df_Turbine_condenser_merge["Matched_Turbine_power_MW_SF"] = matched_power_SF
df_Turbine_condenser_merge["Matched_h2_actual_KJ_Kg"] = matched_h2_actual
df_Turbine_condenser_merge["Power_Error"] = (
    df_Turbine_condenser_merge["Matched_Turbine_power_MW_EE"] -
    df_Turbine_condenser_merge["K1601_Total_estimated_power_MW"]
)
df_Turbine_condenser_merge["Power_EE_vs_SF_Diff"] = (
    df_Turbine_condenser_merge["Matched_Turbine_power_MW_EE"] -
    df_Turbine_condenser_merge["Matched_Turbine_power_MW_SF"]
)
df_Turbine_condenser_merge["Devaiation in steam flow (Simulated-actual)"] = df_Turbine_condenser_merge["Calculated_Steam_flow_TPH"]- df_Turbine_condenser_merge["KT-1601 steam flow"]
df_Turbine_condenser_merge["Devaiation in extraction (Simulated-actual)"] = df_Turbine_condenser_merge["Optimized_Extraction_flow_TPH"]- df_Turbine_condenser_merge["KT-1601 Extraction flow"]
df_Turbine_condenser_merge["Specific_steam_consumption_MT_MW_updated"] = df_Turbine_condenser_merge["Calculated_Steam_flow_TPH"] / df_Turbine_condenser_merge["K1601_Total_estimated_power_MW"]

df_Turbine_condenser_merge = df_Turbine_condenser_merge.round(2)
df_Turbine_condenser_merge.to_excel(file_path_KT1601 +"/"+ "Yan_OLF1_PRC K1601_compressor_turbine_simulated_data.xlsx",freeze_panes=(1,0))

#%% optimization of steam and condensate flow

from scipy.optimize import minimize_scalar

optimized_condensate = []
calculated_steam_flow = []
matched_power_EE = []
matched_power_SF = []
matched_h2_actual = []

def match_actual_power(row):
    try:
        # Steam inlet enthalpy
        P_steam = row["Steam pressure"] * 1000  # Pa
        T_steam = row["Steam Temp"] + 273.15    # K
        h_steam = PropsSI("H", "P", P_steam, "T", T_steam, "Water") / 1000  # kJ/kg
     
        # Actual extracted steam enthalpy (he)
        Pe = row["Extraction Pressure"] * 1000  # Pa
        # Te_actual = row["Extraction Temperature"] + 273.15  # K
        T_sat_extraction = PropsSI('T', 'P', Pe, 'Q', 1, 'Water') # K
        he = PropsSI('H', 'P', Pe, 'T', T_sat_extraction + 0.01, 'Water')/1000  # Extracted steam (actual)
        
        # Actual condenser steam enthalpy (hc)
        Pc = row["Condensate Pressure"] * 1000  # Pa
        # Tc_actual = row["Condensate Temperature"] + 273.15  # K
        T_sat_condensate = PropsSI('T', 'P', Pc, 'Q', 0, 'Water') # degC
        hc_liquid = PropsSI('H', 'P', Pc, 'T', T_sat_condensate - 0.01, 'Water')/1000  # Condenser outlet (actual)
        dryness_fraction = 0.92
        hc_vapor = PropsSI('H', 'P', Pc, 'T', T_sat_condensate + 0.01, 'Water')/1000  # Condenser outlet (actual)
    
        hc = hc_liquid + hc_vapor*dryness_fraction
    
        extraction_flow_TPH = row['KT-1601 Extraction flow']
        extraction_flow_Kg_hr = extraction_flow_TPH * 1000
        actual_power = row["K1601_Total_estimated_power_MW"]
    
        def objective(condensate_flow_TPH):
            condensate_flow_Kg_hr = condensate_flow_TPH * 1000
            steam_flow_Kg_hr = extraction_flow_Kg_hr + condensate_flow_Kg_hr
    
            power = ((extraction_flow_Kg_hr * (h_steam - he)) + 
                     (condensate_flow_Kg_hr * (h_steam - hc))) / 3600 / 1000
            return (power - actual_power) ** 2
    
        result = minimize_scalar(objective, bounds=(10, 350), method='bounded')
    
        if result.success:
            cf_opt = result.x
            cf_Kg_hr = cf_opt * 1000
            sf_opt = cf_opt + extraction_flow_TPH
            sf_Kg_hr = sf_opt * 1000
            # Matched power via energy balance
            power_matched_EE = ((extraction_flow_Kg_hr * (h_steam - he)) + (cf_Kg_hr * (h_steam - hc))) / 3600 / 1000

            # Mass-weighted outlet enthalpy
            h2_actual = (extraction_flow_Kg_hr * he + cf_Kg_hr * hc) / sf_Kg_hr

            # Power via inlet/outlet enthalpy and total flow
            power_matched_SF = (h_steam - h2_actual) * (sf_Kg_hr / 3600 / 1000)

            return cf_opt, sf_opt, power_matched_EE, power_matched_SF, h2_actual
        else:
            return (row["KT-1601 condensate flow"], row["KT-1601 steam flow"],
                    row["Turbine_power_MW_based_on_EE"], row["Turbine_power_MW_based_on_steam_flow"],
                    None)
    except Exception as e:
        print(f"Error at index {row.name}: {e}")
        return (row["KT-1601 condensate flow"], row["KT-1601 steam flow"],
                row["Turbine_power_MW_based_on_EE"], row["Turbine_power_MW_based_on_steam_flow"],
                None)
    
# Loop and apply
for _, row in df_Turbine_condenser_merge.iterrows():
    cf, sf, power_EE, power_SF, h2 = match_actual_power(row)
    optimized_condensate.append(cf)
    calculated_steam_flow.append(sf)
    matched_power_EE.append(power_EE)
    matched_power_SF.append(power_SF)
    matched_h2_actual.append(h2)

# Save to DataFrame
df_Turbine_condenser_merge["Optimized_Condensate_flow_TPH"] = optimized_condensate
df_Turbine_condenser_merge["Calculated_Steam_flow_TPH_varying_condensate"] = calculated_steam_flow
df_Turbine_condenser_merge["Matched_Turbine_power_MW_EE_varying_condensate"] = matched_power_EE
df_Turbine_condenser_merge["Matched_Turbine_power_MW_SF_varying_condensate"] = matched_power_SF
df_Turbine_condenser_merge["Matched_h2_actual_KJ_Kg_varying_condensate"] = matched_h2_actual
df_Turbine_condenser_merge["Power_Error"] = (
    df_Turbine_condenser_merge["Matched_Turbine_power_MW_EE_varying_condensate"] -
    df_Turbine_condenser_merge["K1601_Total_estimated_power_MW"]
)
df_Turbine_condenser_merge["Power_EE_vs_SF_Diff"] = (
    df_Turbine_condenser_merge["Matched_Turbine_power_MW_EE_varying_condensate"] -
    df_Turbine_condenser_merge["Matched_Turbine_power_MW_SF_varying_condensate"]
)
df_Turbine_condenser_merge["Devaiation in steam flow_basedon_varying_condensate(Simulated-actual)"] = df_Turbine_condenser_merge["Calculated_Steam_flow_TPH_varying_condensate"]- df_Turbine_condenser_merge["KT-1601 steam flow"]
df_Turbine_condenser_merge["Devaiation in condensate_flow_basedon_varying_condensate(Simulated-actual)"] = df_Turbine_condenser_merge["Optimized_Condensate_flow_TPH"]- df_Turbine_condenser_merge['KT-1601 condensate flow']
df_Turbine_condenser_merge["Specific_steam_consumption_MT_MW__basedon_varying_condensate"] = df_Turbine_condenser_merge["Calculated_Steam_flow_TPH_varying_condensate"] / df_Turbine_condenser_merge["K1601_Total_estimated_power_MW"]

df_Turbine_condenser_merge = df_Turbine_condenser_merge.round(2)
df_Turbine_condenser_merge.to_excel(file_path_KT1601 +"/"+ "Yan_OLF1_PRC K1601_compressor_turbine_simulated_data.xlsx",freeze_panes=(1,0))

#%% Static KT-1601 simulation 
# Mass flows
Steam_flow_TPH = 54.722
Extraction_flow_TPH = 0
Condensate_flow_TPH = 54.722

# Extraction_flow_TPH = Steam_flow_TPH - Condensate_flow_TPH

Steam_flow_Kg_hr       = Steam_flow_TPH *1000
Extraction_flow_Kg_hr  = Extraction_flow_TPH *1000
Condensate_flow_Kg_hr  = Condensate_flow_TPH *1000

# Steam Inlet conditions
P_steam = 96.6936*1e5     # Pa
T_steam = -16 + 273.15   # K
T_sat_steam = PropsSI('T', 'P', P_steam, 'Q', 1, 'Ammonia') - 273.15 # degC

# Extraction pressure
Pe = 1*1e5         # Pa
Te_actual = 219 + 273.15  # K (actual extracted steam temp) 
T_sat_extraction = PropsSI('T', 'P', Pe, 'Q', 1, 'Ammonia') # degC

# Condenser pressure
Pc = 1*1e5
Tc_actual = -33 + 273.15  # K (actual exhaust steam temp)
T_sat_condensate = PropsSI('T', 'P', Pc, 'Q', 0, 'Ammonia') # degC

# Steam (inlet) Enthalpy calculation 
h_steam = PropsSI('H', 'P', P_steam, 'T', T_steam, 'Ammonia')/1000    # kJ/kg
s_steam = PropsSI('S', 'P', P_steam, 'T', T_steam, 'Ammonia')/1000  # J/kg·K

# Actual outlet enthalpies
#he = PropsSI('H', 'P', Pe, 'T', Te_actual, 'Water')/1000  # Extracted steam (actual)
he = PropsSI('H', 'P', Pe, 'T', T_sat_extraction + 0.01, 'Ammonia')/1000  # Extracted steam (actual)
#hc = PropsSI('H', 'P', Pc, 'T', Tc_actual, 'Water')/1000  # Condenser outlet (actual)
hc_liquid = PropsSI('H', 'P', Pc, 'T', T_sat_condensate - 0.01, 'Ammonia')/1000  # Condenser outlet (actual)
dryness_fraction = 0.92
hc_vapor = PropsSI('H', 'P', Pc, 'T', T_sat_condensate + 0.01, 'Water')/1000  # Condenser outlet (actual)

hc = hc_liquid + hc_vapor*dryness_fraction

# hc = PropsSI('H', 'P', Pc, 'Q', dryness_fraction, 'Water') / 1000
Net_heat_release = Steam_flow_Kg_hr*h_steam -Extraction_flow_Kg_hr*he - Condensate_flow_Kg_hr*hc
power_gen_extraction = Extraction_flow_Kg_hr*(h_steam-he)/3600
power_gen_exhaust = Condensate_flow_Kg_hr*(h_steam-hc)/3600
Turbine_power_based_on_EE = power_gen_extraction + power_gen_exhaust
Specific_steam_consumption_MT_MW = Steam_flow_TPH/(Turbine_power_based_on_EE/1000) # MT/MW

def isentropic_enthalpy(P_target, s_in):
    # Keep everything in J/kg·K
    s_f = PropsSI("S", "P", P_target, "Q", 0, "Water")/1000
    s_g = PropsSI("S", "P", P_target, "Q", 1, "Water")/1000

    if s_f < s_in < s_g:
        x = (s_in - s_f) / (s_g - s_f)
        h_f = PropsSI("H", "P", P_target, "Q", 0, "Water")
        h_g = PropsSI("H", "P", P_target, "Q", 1, "Water")
        h_iso = h_f + x * (h_g - h_f)
    else:
        h_iso = PropsSI("H", "P", P_target, "S", s_in, "Water")

    return h_iso / 1000  # Convert to kJ/kg


# Isentropic outlet enthalpies
he_s = isentropic_enthalpy(Pe, s_steam)
hc_s = isentropic_enthalpy(Pc, s_steam)

# Mass-weighted outlet enthalpies
h2_actual = (Extraction_flow_Kg_hr * he + Condensate_flow_Kg_hr * hc) / Steam_flow_Kg_hr
h2s_ideal  = (Extraction_flow_Kg_hr  * he_s + Condensate_flow_Kg_hr * hc_s) / Steam_flow_Kg_hr

# Turbine isentropic efficiency
efficiency = (h_steam - h2_actual) / (h_steam - h2s_ideal)

turbine_power_KW = (h_steam - h2_actual) * (Steam_flow_Kg_hr / 3600)  # Convert kg/hr to kg/s

# Convert to kJ/kg
print(f"h_steam       = {h_steam:.2f} kJ/kg")
print(f"h2       = {h2_actual:.2f} kJ/kg")
print(f"h2 (iso) = {h2s_ideal:.2f} kJ/kg")
print(f"Efficiency = {efficiency*100:.2f} %")
print(f"Power Output = {turbine_power_KW:.2f} kW")

#%% Extraction flow optimization for KT-1601 static simulation
def turbine_metrics(Extraction_flow_TPH):
    Extraction_flow_Kg_hr = Extraction_flow_TPH * 1000

    # Total outlet enthalpy
    h2_actual = (Extraction_flow_Kg_hr * he + Condensate_flow_Kg_hr * hc) / Steam_flow_Kg_hr
    h2s_ideal = (Extraction_flow_Kg_hr * he_s + Condensate_flow_Kg_hr * hc_s) / Steam_flow_Kg_hr

    # Efficiency
    efficiency = (h_steam - h2_actual) / (h_steam - h2s_ideal)

    # Power
    turbine_power_kW = (h_steam - h2_actual) * Steam_flow_Kg_hr / 3600

    return efficiency, turbine_power_kW


def constraint_check(Extraction_flow_TPH):
    efficiency, power = turbine_metrics(Extraction_flow_TPH)

    # Check if within hard constraints
    if 0.75 <= efficiency <= 0.85 and power < 40000:
        return 0  # No penalty if all constraints are met

    # Apply penalty based on distance from ideal efficiency and limit on power
    penalty = 1e6
    if not (0.75 <= efficiency <= 0.85):
        penalty += 1e5 * abs(efficiency - 0.80)  # Ideal center efficiency = 80%
    if power >= 40000:
        penalty += 1e3 * (power - 40000)  # Penalize excess power more sharply

    return penalty


# Search bounds (can't extract more than total steam or less than zero)
result = minimize_scalar(
    constraint_check,
    bounds=(0, Steam_flow_TPH), 
    method='bounded'
)

# Best extraction flow that meets constraint
import matplotlib.pyplot as plt

extraction_range = np.linspace(0, Steam_flow_TPH, 100)
eff_list = []
power_list = []

for ext in extraction_range:
    eff, power = turbine_metrics(ext)
    eff_list.append(eff * 100)  # Convert to %
    power_list.append(power)   # in kW

fig, ax1 = plt.subplots(figsize=(10, 5))

# Power on left y-axis
ax1.set_xlabel('Extraction Flow (TPH)')
ax1.set_ylabel('Turbine Power (kW)', color='tab:blue')
ax1.plot(extraction_range, power_list, color='tab:blue', label='Power')
ax1.axhline(150000, color='r', linestyle='--', label='Power Limit')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Twin axis for efficiency
ax2 = ax1.twinx()
ax2.set_ylabel('Efficiency (%)', color='tab:green')
ax2.plot(extraction_range, eff_list, color='tab:green', label='Efficiency')
ax2.tick_params(axis='y', labelcolor='tab:green')

fig.suptitle('Turbine Performance vs Extraction Flow')
fig.tight_layout()
plt.grid(True)
plt.show()