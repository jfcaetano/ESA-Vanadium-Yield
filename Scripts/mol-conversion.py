### Mol Conversion
### JFCAETANO 2023
### MIT Licence

import rdkit, sys, time, csv, math
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np


# Setup dataframe
data_filename = 'Database_0.csv'
output_fn = 'Database_1.csv'

my_descriptors = list()
for desc_name in dir(Descriptors):
    if desc_name in ['BalabanJ','BertzCT','TPSA']:
        my_descriptors.append(desc_name)
    elif desc_name[:3]=='Chi':
        my_descriptors.append(desc_name)
    elif 'VSA' in desc_name:
        my_descriptors.append(desc_name)
    elif 'Kappa' in desc_name:
        my_descriptors.append(desc_name)
    elif desc_name[:1]=='H':
        my_descriptors.append(desc_name)
    elif desc_name[:1]=='N':
        my_descriptors.append(desc_name)
    elif desc_name[:1]=='M':
        my_descriptors.append(desc_name)

###


# Prepare calculations
f = open(data_filename,'r')
reader = csv.DictReader(f, delimiter=',')
#
o = list()
for row in reader:
    # Columns to maintain
    nl = dict()
    nl['Entry']                      = row['Entry']
    nl['Group']                      = row['Group']
    nl['Cat_Structure']              = row['Cat_Structure']
    nl['Cat_Group']                  = row['Cat_Group']
    nl['Catalyst']                   = row['Catalyst']
    nl['Substrate']                  = row['Substrate']
    nl['Ligand']                     = row['Ligand']
    nl['Oxidant']                    = row['Oxidant']
    nl['Oxidant_Group']              = row['Oxidant_Group']
    nl['Solvent']                    = row['Solvent']
    nl['Solvent_Group']              = row['Solvent_Group']
    nl['Solv_opt_freq20']            = row['Solv_opt_freq20']
    nl['Solv_opt_freq25']            = row['Solv_opt_freq25']
    nl['Solv_Hbond_ac']              = row['Solv_Hbond_ac'] 
    nl['Solv_Hbond_bs']              = row['Solv_Hbond_bs']
    nl['Solv_surf_tens']             = row['Solv_surf_tens']
    nl['Solv_diele_cst']             = row['Solv_diele_cst']
    nl['Solv_aromcity']              = row['Solv_aromcity']
    nl['Solv_elect_halo']            = row['Solv_elect_halo']
    nl['Temp_K']                     = row['Temp_K'] 
    nl['Yield']                      = row['Yield']	
    nl['EE']                         = row['EE']	
    nl['Configuration']              = row['Configuration']	
    nl['Substrate_quant_mmol']       = row['Substrate_quant_mmol']	
    nl['Catalyst_quant_mmol']        = row['Catalyst_quant_mmol']
    nl['Ligand_quant_mmol']          = row['Ligand_quant_mmol']
    nl['Oxidant_quant_mmol']         = row['Oxidant_quant_mmol']
    nl['Additive_quant_mL']          = row['Additive_quant_mL']
    nl['Solution_vol_mL']            = row['Solution_vol_mL']
    nl['Time_h']                     = row['Time_h']

    
    # Load Compound
    comp_cat = row['Catalyst']
    comp_cat_s = Chem.MolFromMolFile(f"{comp_cat}.mol")
    comp_sub = row['Substrate']
    comp_sub_s = Chem.MolFromMolFile(f"{comp_sub}.mol")
    comp_sol = row['Solvent']
    comp_sol_s = Chem.MolFromMolFile(f"{comp_sol}.mol")
    comp_lig = row['Ligand']
    if comp_lig not in ['NA']:
        comp_lig_s = Chem.MolFromMolFile(f"{comp_lig}.mol")


    # Calculate Catal Descriptors
    for desc in my_descriptors:
        nl[f"{desc}_Cat"]=eval(f"Descriptors.{desc}(comp_cat_s)")
        nl[f"{desc}_Sub"]=eval(f"Descriptors.{desc}(comp_sub_s)")
        nl[f"{desc}_Sol"]=eval(f"Descriptors.{desc}(comp_sol_s)")
        nl[f"{desc}_Lig"]=eval(f"Descriptors.{desc}(comp_lig_s)")
    # Append nl to output list
    
    o.append(nl)


with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)

# Clean up stuff
f.close()
