df_test = pd.read_csv('testing_data.csv')

for col, miss_val in column_to_missing.items():
    df_test[col] = df_test[col].replace(miss_val, np.nan)
message = "True Missing Counts:"
print(message, "\n", "="*len(message), sep='')
df_test.isnull().sum()

for i in range(1,4):

    mask = df_test[f'diag_{i}'].str.contains('^[EV]', regex=True, na=False)
    mask_nan = df_test[f'diag_{i}'].isna()
    combined_mask = mask | mask_nan
    
    df_test.loc[~combined_mask, f'diag_{i}'] = df_test.loc[~combined_mask, f'diag_{i}'].apply(categorize_diag)
    df_test.loc[mask, f'diag_{i}'] = df_test.loc[mask, f'diag_{i}'].apply(convert_EV_code)

"""
Based on medical specialty => reduces total features

Applies to: `medical_specialty`
"""
df_test['medical_specialty'] = df_test['medical_specialty'].apply(map_specialty_to_group)
print("medical_specialty: ", df_test['medical_specialty'].unique(), end = "\n\n")

"""
Categorizing ambiguous mappings together.

For example, admission_source_id has categories 9, 15, 17, 20, and 21, which map to
Not availble, not availible, Null, not mapped, unknown

Group 9, 15, 17, 20 and 21 together to null. Reduces features & remove noise

Applies to `admission_type_id`, `discharge_disposition_id`, `admission_source_id`
"""
df_test['admission_type_id'] = df_test['admission_type_id'].apply(build_map_to_none([5, 6, 8]))
df_test['discharge_disposition_id'] = df_test['discharge_disposition_id'].apply(build_map_to_none([18, 25, 26]))
df_test['admission_source_id'] = df_test['admission_source_id'].apply(build_map_to_none([9, 15, 17, 20, 21]))

print("admission_type_id: ", np.sort(df_test['admission_type_id'].unique()), end = "\n\n")
print("discharge_disposition_id: ", np.sort(df_test['discharge_disposition_id'].unique()), end = "\n\n")
print("admission_source_id: ", np.sort(df_test['admission_source_id'].unique()), end = "\n\n")

df_test.drop(columns=['weight', 'examide', 'citoglipton'], inplace=True)

mode_impute_col_names = ['race', 'payer_code', 'medical_specialty', 'gender', 'diag_1', 'diag_2', 'diag_3', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']

def mode_impute(df_column):
    mask = df_column.notna()
    mode = df_column[mask].mode()[0]
    print(f"Mode:\t\t", mode)
    updated_df_column = df_column.fillna(mode)
    return updated_df_column

for name in mode_impute_col_names:
    print(name, end=' ')
    df_test[name] = mode_impute(df_test[name])

col_names = ['max_glu_serum', 'A1Cresult']

for name in col_names:
    df_test[name] = df_test[name].replace(np.nan, 'Unknown')