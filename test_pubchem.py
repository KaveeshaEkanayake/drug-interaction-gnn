import pandas as pd
import requests

df = pd.read_csv('data/raw/db_drug_interactions.csv')
df.columns = ['drug1_name', 'drug2_name', 'description']

print('Sample drug names:')
print(df['drug1_name'].unique()[:10].tolist())
print(df['drug2_name'].unique()[:10].tolist())

print('\nTesting PubChem with Aspirin...')
try:
    r = requests.get(
        'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/Aspirin/property/CanonicalSMILES/JSON',
        timeout=10
    )
    print('Status:', r.status_code)
    print('Response:', r.text[:200])
except Exception as e:
    print('Error:', e)