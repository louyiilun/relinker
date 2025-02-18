import argparse
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
from tqdm import tqdm
from pdb import set_trace
import data.zinc.prepare_dataset as prep
import multiprocessing as mp
import sys
sys.path.append('../../')
from src.datasets import read_sdf
import subprocess


def load_rdkit_molecule(xyz_path, obabel_path, true_frag_smi):
    if not os.path.exists(obabel_path):
        subprocess.run(f'obabel {xyz_path} -O {obabel_path}', shell=True)

    supp = Chem.SDMolSupplier(obabel_path, sanitize=False)
    mol = list(supp)[0]

    # Keeping only the biggest connected part
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    mol_filtered = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    try:
        mol_smi = Chem.MolToSmiles(mol_filtered)
    except RuntimeError:
        mol_smi = Chem.MolToSmiles(mol_filtered, canonical=False)

    # Retrieving linker
    true_frag = Chem.MolFromSmiles(true_frag_smi, sanitize=False)
    match = mol_filtered.GetSubstructMatch(true_frag)
    if len(match) == 0:
        linker_smi = ''
    else:
        elinker = Chem.EditableMol(mol_filtered)
        for atom in sorted(match, reverse=True):
            elinker.RemoveAtom(atom)
        linker = elinker.GetMol()
        Chem.Kekulize(linker, clearAromaticFlags=True)
        try:
            linker_smi = Chem.MolToSmiles(linker)
        except RuntimeError:
            linker_smi = Chem.MolToSmiles(linker, canonical=False)

    return mol_filtered, mol_smi, linker_smi


def load_molecules(folder, true_frag_smi):
    obabel_dir = f'{folder}/obabel'
    os.makedirs(obabel_dir, exist_ok=True)

    pred_mols = []
    pred_mols_smi = []
    pred_link_smi = []
    for fname in os.listdir(folder):
        number = fname.split('_')[0]
        if number.isdigit():
            pred_path = f'{folder}/{fname}'
            pred_obabel_path = f'{obabel_dir}/{number}_.sdf'
            mol, mol_smi, link_smi = load_rdkit_molecule(pred_path, pred_obabel_path, true_frag_smi)
            pred_mols.append(mol)
            pred_mols_smi.append(mol_smi)
            pred_link_smi.append(link_smi)

    return pred_mols, pred_mols_smi, pred_link_smi


def load_sampled_dataset(folder, idx2true_mol_smi, idx2true_frag_smi):
    pred_mols = []
    pred_mols_smi = []
    pred_link_smi = []
    true_mols_smi = []
    true_frags_smi = []

    for fname in tqdm(os.listdir(folder)):
        if fname.isdigit():
            true_mol_smi = idx2true_mol_smi[int(fname)]
            true_frag_smi = idx2true_frag_smi[int(fname)]

            mols, mols_smi, link_smi = load_molecules(f'{folder}/{fname}', true_frag_smi)
            pred_mols += mols
            pred_mols_smi += mols_smi
            pred_link_smi += link_smi
            true_mols_smi += [true_mol_smi] * len(mols)
            true_frags_smi += [true_frag_smi] * len(mols)

    return pred_mols, pred_mols_smi, pred_link_smi, true_mols_smi, true_frags_smi


def reformat(samples, true_smiles_path):
    true_smiles_path = os.path.join(true_smiles_path)

    reformat_smi = []
# 读取true_smiles_path路径的文件，获取真实分子和碎片的SMILES字符串
    true_smiles_table = pd.read_csv(true_smiles_path, sep=' ', names=['molecule', 'fragments'])
    idx2true_mol_smi = dict(enumerate(true_smiles_table.molecule.values))
    idx2true_frag_smi = dict(enumerate(true_smiles_table.fragments.values))
# 调用load_sampled_dataset函数，获取预测分子列表、预测分子的SMILES字符串列表、预测连接器的SMILES字符串列表、真实分子的SMILES字符串列表和真实碎片的SMILES字符串列表
    pred_mols, pred_mols_smi, pred_link_smi, true_mols_smi, true_frag_smi = load_sampled_dataset(
        folder=samples,
        idx2true_mol_smi=idx2true_mol_smi,
        idx2true_frag_smi=idx2true_frag_smi,
    )
    # print(f"pred_mols_num = {len(pred_mols)}")
    # print(pred_mols_smi)

    logplist = []

    for m in pred_mols:
        # try:
            Chem.SanitizeMol(m)
            Chem.AssignStereochemistry(m, force=True, cleanIt=True)
            mol = Chem.AddHs(m)
            logp = Crippen.MolLogP(mol)
            logplist.append(logp)
        # except:
        #     continue

    # print(f"logplist = {len(logplist)}")
    
    Clogp_list = []
    goal_ClogP =2.5

    for logp in logplist:
        clogp = max(0.0, 1 - (1/6)*abs(logp - goal_ClogP))
        Clogp_list.append(clogp)
    rclogp = sum(Clogp_list) / len(Clogp_list)
    print(rclogp)

    os.makedirs(samples,exist_ok=True)
    rclogp_value_path = os.path.join(samples, 'rclogp_value.txt')

    with open(rclogp_value_path, "w") as file:
        file.write(str(rclogp))

    for i in range(len(pred_mols)):
        reformat_smi.append(f'{pred_mols_smi[i]} {pred_link_smi[i]} {true_frag_smi[i]}')
        sdf_filename = f'reformat_mol_{i}.sdf'# 我是搁这儿来的
        sdf_path = os.path.join(samples, sdf_filename)
        with Chem.SDWriter(open(sdf_path, 'w')) as writer:
            writer.write(pred_mols[i]) 
        
    return reformat_smi #我可以拿这个来生成，我试一下

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--samples', action='store', type=str, required=True) # input
    # parser.add_argument('--out_dir', action='store', type=str, required=True) # output
    # parser.add_argument('--true_smiles_path', action='store', type=str, required=True) # true molecular
    # parser.add_argument('--template', action='store', type=str, required=True)
    # parser.add_argument('--n_samples', action='store', type=int, required=True)
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', action='store', type=str, required=False, default = "DiffLinker/sample1") # input
    parser.add_argument('--out_dir', action='store', type=str, required=False, default = "DiffLinker/out_dir") # output
    parser.add_argument('--true_smiles_path', action='store', type=str, required=False, default = "DiffLinker/datasets/zinc_final_test_smiles.smi") # true molecular
    parser.add_argument('--template', action='store', type=str, required=False, default = "reformat_mol")
    parser.add_argument('--n_samples', action='store', type=int, required=False, default = 2)
    args = parser.parse_args()

    reformat_smi = reformat(
        samples=args.samples,
        true_smiles_path=args.true_smiles_path,
    )

    processes = []
    for pid in range(args.n_samples):
    # for pid :
        sdf_path = os.path.join(args.samples, f'reformat_mol_{pid}.sdf')
        out_mol_path = os.path.join(args.out_dir, f'{args.template}_mol_{pid}.sdf')
        out_frag_path = os.path.join(args.out_dir, f'{args.template}_frag_{pid}.sdf')
        out_link_path = os.path.join(args.out_dir, f'{args.template}_link_{pid}.sdf')
        out_table_path = os.path.join(args.out_dir, f'{args.template}_table_{pid}.csv')
        kwargs = {
            'table_smi': reformat_smi,
            'sdf_path': sdf_path,
            'out_mol_path': out_mol_path,
            'out_frag_path': out_frag_path,
            'out_link_path': out_link_path,
            'out_table_path': out_table_path,
            'progress': pid == 0,
        }
        os.makedirs(args.out_dir, exist_ok=True)
        process = mp.Process(target=prep.run, kwargs=kwargs)
        process.start()
        processes.append(process)

    for pid in range(args.n_samples):
        processes[pid].join()



