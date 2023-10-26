import json
from tqdm import tqdm

with open("./data/FEVER/processed/test.json", 'r') as f:
    test_data = json.load(f)
id_evidence = {}
for data in test_data:
    id_evidence[data['id']] = data['evidence']
print(len(id_evidence))

with open("./results/programs/FEVER_test_N1_aquilacode-7b-nv_programs_with_evidence.json", 'r') as f:
    generated_programs = json.load(f)
print(len(generated_programs))

processed_data = []
for data in generated_programs:
    print(data['id'])
    processed_data.append({
        'idx': data['idx'],
        'id': data['id'],
        'claim': data['claim'],
        'label': data['label'],
        'evidence': id_evidence[data['id']],
        'predicted_program': data['predicted_program']
    })

save_path = "./results/programs/FEVER_test_N1_aquilacode-7b-nv_programs_with_evidence_new.json"
with open(save_path, 'w') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)