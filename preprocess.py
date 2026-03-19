import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


df_test = pd.read_csv('testing_data.csv')

missing_cols = ['race', 'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'gender']


column_to_missing = {}
for col in missing_cols:
    if col != 'gender':
        column_to_missing[col] = '?'
    else:
        column_to_missing[col] = 'Unknown/Invalid'

for col, miss_val in column_to_missing.items():
    df_test[col] = df_test[col].replace(miss_val, np.nan)

"""
Diagnosis Groups:

Grouping based on ICD9 Grouping (source: Wikipedia)
"""
diags_ICD9_dict = {
    '001–139': 'infectious and parasitic diseases',
    '140–239': 'neoplasms',
    '240–279': 'endocrine, nutritional and metabolic diseases, and immunity disorders',
    '280–289': 'diseases of the blood and blood-forming organs',
    '290–319': 'mental disorders',
    '320–389': 'diseases of the nervous system and sense organs',
    '390–459': 'diseases of the circulatory system',
    '460–519': 'diseases of the respiratory system',
    '520–579': 'diseases of the digestive system',
    '580–629': 'diseases of the genitourinary system',
    '630–679': 'complications of pregnancy, childbirth, and the puerperium',
    '680–709': 'diseases of the skin and subcutaneous tissue',
    '710–739': 'diseases of the musculoskeletal system and connective tissue',
    '740–759': 'congenital anomalies',
    '760–779': 'certain conditions originating in the perinatal period',
    '780–799': 'symptoms, signs, and ill-defined conditions',
    '800–999': 'injury and poisoning',
    'E & V codes': 'external causes of injury and supplemental classification',
}

def categorize_diag(code: str) -> str:
    code_float = float(code)
    code_int = int(code_float)
    if code_int >= 800:
        return '800–999'
    elif code_int >= 780:
        return '780–799'
    elif code_int >= 760:
        return '760–779'
    elif code_int >= 740:
        return '740–759'
    elif code_int >= 710:
        return '710–739'
    elif code_int >= 680:
        return '680–709'
    elif code_int >= 630:
        return '630–679'
    elif code_int >= 580:
        return '580–629'
    elif code_int >= 520:
        return '520–579'
    elif code_int >= 460:
        return '460–519'
    elif code_int >= 390:
        return '390–459'
    elif code_int >= 320:
        return '320–389'
    elif code_int >= 290:
        return '290–319'
    elif code_int >= 280:
        return '280–289'
    elif code_int >= 240:
        return '240–279'
    elif code_int >= 140:
        return '140–239'
    else:
        return '001–139'

def convert_EV_code(code: str) -> str:
    return 'E & V codes'


"""
Medical Specialty Groups:

Grouping based on Google Gemini output
"""

specialty_mapping = {
    'PrimaryCare': [
        'Family/GeneralPractice', 'InternalMedicine', 'Hospitalist', 
        'Osteopath', 'Resident'
    ],
    'Surgery': [
        'Surgery-General', 'Surgery-Vascular', 'Urology', 'Surgery-Neuro', 
        'Orthopedics-Reconstructive', 'Surgery-Cardiovascular/Thoracic', 
        'Orthopedics', 'Surgery-Plastic', 'Podiatry', 'Surgery-Thoracic', 
        'Surgery-Colon&Rectal', 'Otolaryngology', 'Surgery-Cardiovascular', 
        'Surgeon', 'Proctology', 'Surgery-Maxillofacial', 'SurgicalSpecialty', 
        'Dentistry', 'Ophthalmology', 'Surgery-PlasticwithinHeadandNeck'
    ],
    'InternalMedicine_Subspecialty': [
        'Cardiology', 'Gastroenterology', 'Nephrology', 'Pulmonology', 
        'Neurology', 'Hematology/Oncology', 'Endocrinology-Metabolism', 
        'Hematology', 'Oncology', 'Endocrinology', 'InfectiousDiseases', 
        'Rheumatology', 'AllergyandImmunology', 'Neurophysiology', 'Dermatology'
    ],
    'Emergency_CriticalCare': [
        'Emergency/Trauma'
    ],
    'Maternal_Pediatric': [
        'ObstetricsandGynecology', 'Pediatrics-CriticalCare', 'Pediatrics-Endocrinology', 
        'Pediatrics', 'Gynecology', 'Pediatrics-Pulmonology', 'Pediatrics-Neurology', 
        'Obsterics&Gynecology-GynecologicOnco', 'Pediatrics-AllergyandImmunology', 
        'Cardiology-Pediatric', 'Anesthesiology-Pediatric', 'Obstetrics', 
        'Surgery-Pediatric', 'Perinatology', 'Pediatrics-EmergencyMedicine', 
        'Pediatrics-Hematology-Oncology', 'Psychiatry-Child/Adolescent', 
        'Pediatrics-InfectiousDiseases'
    ],
    'Psych_Rehab_Support': [
        'PhysicalMedicineandRehabilitation', 'Psychiatry', 'Psychology', 
        'Anesthesiology', 'Speech', 'SportsMedicine'
    ],
    'Diagnostics_Other': [
        'Radiologist', 'Radiology', 'Pathology', 'OutreachServices', 'DCPTEAM'
    ],
    'Missing': [
        'PhysicianNotFound'
    ]
}


quad_med = ['Down', 'No', 'Steady', 'Up']
binary_med = ['No', 'Steady']

mapping = {
    # Custom/Unique orderings
    'gender': ['Male', 'Female'],
    'age': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
    'weight': ['[0-25)', '[25-50)', '[50-75)', '[75-100)', '[100-125)', '[125-150)', '[150-175)', '[175-200)', '>200'],
    'max_glu_serum': ['Unknown', 'Norm', '>200', '>300'],
    'A1Cresult': ['Unknown', 'Norm', '>7', '>8'],
    'change': ['No', 'Ch'],
    'diabetesMed': ['No', 'Yes'],
    'readmitted': ['NO', '>30', '<30'],
    
    # 3-level medication (Unique)
    'tolazamide': ['No', 'Steady', 'Up'],
    
    # 4-level medications (Repeated)
    'metformin': quad_med,
    'repaglinide': quad_med,
    'nateglinide': quad_med,
    'chlorpropamide': quad_med,
    'glimepiride': quad_med,
    'glipizide': quad_med,
    'glyburide': quad_med,
    'pioglitazone': quad_med,
    'rosiglitazone': quad_med,
    'acarbose': quad_med,
    'miglitol': quad_med,
    'insulin': quad_med,
    'glyburide-metformin': quad_med,
    
    # 2-level medications (Repeated)
    'acetohexamide': binary_med,
    'tolbutamide': binary_med,
    'troglitazone': binary_med,
    'glipizide-metformin': binary_med,
    'glimepiride-pioglitazone': binary_med,
    'metformin-rosiglitazone': binary_med,
    'metformin-pioglitazone': binary_med
}

def map_specialty_to_group(specialty):
    if pd.isna(specialty):
        return specialty
    for key, value in specialty_mapping.items():
        if specialty in value:
            return key
    return 'Other'

def build_map_to_none(categories : list[int]):
    def map_to_one(category):
        if category in categories:
            return np.nan
        return category
    return map_to_one

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

df_test.drop(columns=['weight', 'examide', 'citoglipton'], inplace=True)

mode_impute_col_names = ['race', 'payer_code', 'medical_specialty', 'gender', 'diag_1', 'diag_2', 'diag_3', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']

def mode_impute(df_column):
    mask = df_column.notna()
    mode = df_column[mask].mode()[0]
    updated_df_column = df_column.fillna(mode)
    return updated_df_column

for name in mode_impute_col_names:
    df_test[name] = mode_impute(df_test[name])

col_names = ['max_glu_serum', 'A1Cresult']

for name in col_names:
    df_test[name] = df_test[name].replace(np.nan, 'Unknown')

df_non_numeric = df_test.select_dtypes(exclude='number')
categorical_columns = list(df_non_numeric.columns.values)
categorical_columns += ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

non_ordinal_cols = [categorical_columns[0]] + categorical_columns[3:8] + categorical_columns[33:]
ordinal_cols = categorical_columns[1:3] + categorical_columns[8:33]

# Ordinal Encoding
for col in ordinal_cols:
    ord_enc = OrdinalEncoder(categories=[mapping[col]])
    df_test[col] = ord_enc.fit_transform(df_test[[col]])

# Non-Ordinal, One-Hot Encoding
df_test = pd.get_dummies(df_test, columns=non_ordinal_cols, drop_first=True)

df_test['readmitted'] = np.select(
    [
        df_test['readmitted_NO'] == True,
        df_test['readmitted_>30'] == False
    ],
    [
        'NO',
        '>30'
    ],
    default='<30'
)

# Drop the extra columns created from encoding
df_test = df_test.drop(columns=["readmitted_NO", "readmitted_>30"])

df_test.to_csv("testing_data_processed.csv")