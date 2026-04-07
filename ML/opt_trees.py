import pandas as pd
import os
import numpy as np
import sys
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from math import sqrt
import time
import argparse


def process_df(df, feature_code):
    # feature_code: [old/new/no]-pol_[old/new/no]-pi_[y/n]-ecfp_[all/no/x/y/z]-torsion_[all/no/x/y/z]-sd
    pol, pi, ecfp = feature_code.split('_')[:-2]
    torsion_selection = feature_code.split('_')[-2]
    sd_selection = feature_code.split('_')[-1]
    
    geo_options = ['x', 'y', 'z']
    torsion_options = ['Total Torsion Angle', 'Mean Torsion Angle', 'Inner Mean Torsion Angle']
    sd_options = ['SD Torsions', 'SD Difference']
    
    geo_select_dict = {
        'all-torsion': [f"{t_opt}_{g_opt}" for t_opt in torsion_options for g_opt in geo_options],
        'all-sd': [f"{s_opt}_{g_opt}" for s_opt in sd_options for g_opt in geo_options],
        'no-torsion': [],
        'no-sd': [],
        'x-torsion': [f"{t_opt}_z" for t_opt in torsion_options],
        'y-torsion': [f"{t_opt}_z" for t_opt in torsion_options],
        'z-torsion': [f"{t_opt}_z" for t_opt in torsion_options],
        'x-sd': [f"{s_opt}_z" for s_opt in sd_options],
        'y-sd': [f"{s_opt}_z" for s_opt in sd_options],
        'z-sd': [f"{s_opt}_z" for s_opt in sd_options],
    }
    
    others_select_dict = {
        'old-pol': ['Molecule Polarizability', 'Average Monomer Polarizability', 'Dipole Moment'],
        'new-pol': ['Rotated Polarizability', 'Average Monomer Polarizability', 'Dipole Moment', 'α_xx', 'α_xy', 'α_yy'],
        'old-pi': ['PiSystemSize'],
        'new-pi': ['PiSystemSize', 'PiSystemArea', 'PiSystemLength'],
        'y-ecfp': [f"ECFP_{i}" for i in range(2048)],
        'no-pi': [],
        'no-pol': [],
        'n-ecfp': []
    }
    
    first_cols = ['Monomer 1', 'Monomer 2', 'Oligomer Type']
    
    pol_cols = others_select_dict.get(pol)
    torsion_cols = geo_select_dict.get(torsion_selection)
    sd_cols = geo_select_dict.get(sd_selection)
    pi_cols = others_select_dict.get(pi)
    ecfp_cols = others_select_dict.get(ecfp)
    spec_cols = ['DFT Oscillator Strength', 'DFT CD Intensity']
    
    if pol.split('-')[0] != 'no' and pi.split('-')[0] != 'no':
        energy_cols = [
            'Molecule HOMO (eV)', 'Molecule LUMO (eV)', 'Molecule HLG (eV)', 
            'ΔE_HOMO (eV)', 'ΔE_LUMO (eV)', 'Monomer HLG (eV)'
        ]
        first_cols.extend(energy_cols)

    col_order = first_cols + pol_cols + torsion_cols + sd_cols + pi_cols + ecfp_cols + spec_cols
    
    df = df[col_order]
    
    df = df.rename(columns = {
        'DFT CD Intensity': 'DFT CD Intensity (a.u.)',
        'DFT Oscillator Strength': 'DFT Oscillator Strength (a.u.)',
    })
    
    new_col_names = []
    for col in df.columns:
        if 'Torsion' in col or 'SD' in col:
            new_col_names.append(col + ' (°)')
        elif 'Polarizability' in col or 'α' in col:
            new_col_names.append(col + ' (Å$^3$)')
        elif 'Dipole' in col:
            new_col_names.append(col + ' (Debye)')
        else:
            new_col_names.append(col)
    
    df.columns = new_col_names
    
    if len(torsion_selection) < 2 and len(sd_selection) < 2:
        df = df.rename(columns = {
            'Total Torsion Angle_z (°)': 'Total Torsion Angle (°)',
            'Mean Torsion Angle_z (°)': 'Mean Torsion Angle (°)',
            'Inner Mean Torsion Angle_z (°)': 'Inner Mean Torsion Angle (°)',
            'SD Torsions_x (°)': 'SD Torsions (°)',
            'SD Difference_x (°)': 'SD Difference (°)'
        })
        
    df = df.dropna().reset_index(drop = True)
    print(f'Number of features (m): {len(df.columns)}')
    print(f'Number of data points (n): {len(df)}')
    print(f"Input features: {list(dict.fromkeys([x.split('_')[0] for x in df.drop(['DFT CD Intensity (a.u.)'], axis = 1).columns]))}")
    return df



def optimize_trees(data_df, rand_state):
    X = data_df.drop(columns = ['DFT CD Intensity (a.u.)'])
    y = data_df['DFT CD Intensity (a.u.)']
 
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state = rand_state)
    n_cols = len(X.columns)

    results_lst = []

    for trees in [10,25,50,75,100,125,150,175,200,500,750,1000,1250,1500]:
        result_dict = {}
        regr = RandomForestRegressor(n_estimators = trees, max_features = None, min_samples_split=2, random_state=0, verbose = 0)
        start_time = time.time() 
        regr.fit(train_X, train_y)
        end_time = time.time()
        runtime = end_time - start_time
        score = regr.score(test_X, test_y)

        test_y_predictions = np.asarray(regr.predict(test_X))
        pdf = test_X[['Monomer 1','Monomer 2','Oligomer Type']].copy(deep=True)
        pdf['DFT CD Intensity'] = np.asarray(test_y)
        pdf["Trained CD Intensity"] =  [x for x in list(test_y_predictions)]

        slope, intercept, r_value, p_value, std_err = linregress(
            pdf['Trained CD Intensity'].astype(float),
            pdf['DFT CD Intensity']
        )
        rmse = sqrt(mean_squared_error(pdf['Trained CD Intensity'],pdf['DFT CD Intensity']))

        result_dict['Trees'] = trees
        result_dict['Score'] = score
        result_dict['TotalRMSE'] = rmse
        result_dict['TotalR2'] = r_value ** 2
        result_dict['Runtime'] = runtime

        results_lst.append(result_dict)
        
    return results_lst



def main(feature_code, rand_state, output_name):
    print(f"{feature_code=}, {rand_state=}")
    rand_state = int(rand_state)
    data = pd.read_csv('clean_dft_df_4096.csv')
    data_df = process_df(data, feature_code)
    if '_ptonly' in output_name:
        data_df = data_df.drop(columns = ['DFT Oscillator Strength (a.u.)'])
        data_df = data_df[data_df['Oligomer Type'] == 2].reset_index(drop = True)
    
    print(list(dict.fromkeys(x for x in data_df.columns)))
    
    results_lst = optimize_trees(data_df, rand_state)
    results_df = pd.DataFrame(results_lst)
    results_df.to_csv(f"{output_name}_{rand_state}.csv", index = False)



ft_dict = {
    'old-pol_old-pi_y-ecfp_all-torsion_all-sd': 'old-old-y-aa',
    'old-pol_old-pi_y-ecfp_z-torsion_x-sd': 'old-old-y-zx',
    'new-pol_old-pi_y-ecfp_z-torsion_x-sd': 'new-old-y-zx',
    'new-pol_new-pi_y-ecfp_z-torsion_x-sd': 'new-new-y-zx',
    
    'old-pol_old-pi_y-ecfp_z-torsion_x-sd': 'old-old-y-zx_ptonly',
}


ft_codes = list(ft_dict.keys())


for i in range(len(ft_codes)):
    ft_code = ft_codes[i]
    output_name = ft_dict.get(ft_code)
    main(ft_code, 42, output_name)
