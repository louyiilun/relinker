import argparse
import os
import os.path as op
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, AllChem, PandasTools, QED, Descriptors, rdMolDescriptors
from tqdm import tqdm
from pdb import set_trace
import data.zinc.prepare_dataset as prep
import multiprocessing as mp
import sys
sys.path.append('../../')
from src.datasets import read_sdf
import subprocess
from multiprocessing import Process
from optparse import OptionParser
import math
import pickle

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
    # print(f"match = {match}")
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
        # print(f'fname = {fname}')
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

def bond_sort_key(bond):
    return bond.GetIdx()

def add_anchor(mol_smi, frag_smi, link_smi):

    frag = Chem.MolFromSmiles(frag_smi)
    link = Chem.MolFromSmiles(link_smi)
    mol = Chem.MolFromSmiles(mol_smi)

    bonds = list(mol.GetBonds())
    matches1 = mol.GetSubstructMatches(frag)
    match1 = []
    match2 = []

    matches2 = mol.GetSubstructMatches(link)
    # print(f"matches= {matches1, matches2}")
    for match_1 in matches1:
        for match_2 in matches2:
            if not set(match_1).intersection(set(match_2)):
                match1.extend(match_1)
                match2.extend(match_2)

    efrag = Chem.EditableMol(frag)
    elink = Chem.EditableMol(link)

    for bond in sorted(bonds, key=bond_sort_key, reverse=True):
        begin_atom = bond.GetBeginAtom() 
        end_atom = bond.GetEndAtom()
        anchor = Chem.Atom('*')

        if begin_atom.GetIdx() in match1 and end_atom.GetIdx() in match2:

            anchor11_idx = begin_atom.GetIdx()
            anchor11 = match1.index(anchor11_idx)
            # print(f"atom11_idx= {anchor11_idx}")
            idx11 = efrag.AddAtom(anchor)
            efrag.AddBond(anchor11, idx11, Chem.rdchem.BondType.SINGLE)
        
            anchor12_idx = end_atom.GetIdx()
            anchor12 = match2.index(anchor12_idx)
            # print(f"atom12_idx= {anchor12_idx}")
            idx12 = elink.AddAtom(anchor)
            elink.AddBond(anchor12, idx12, Chem.rdchem.BondType.SINGLE)
            
        if begin_atom.GetIdx() in match2 and end_atom.GetIdx() in match1:

            anchor21_idx = begin_atom.GetIdx()
            anchor21 = match2.index(anchor21_idx)
            # print(f"atom21_idx= {anchor21_idx}")
            idx21 = elink.AddAtom(anchor)
            elink.AddBond(anchor21, idx21, Chem.rdchem.BondType.SINGLE)
        
            anchor22_idx = end_atom.GetIdx()
            anchor22 = match1.index(anchor22_idx)
            # print(f"atom22_idx= {anchor22_idx}")
            idx22 = efrag.AddAtom(anchor)
            efrag.AddBond(anchor22, idx22, Chem.rdchem.BondType.SINGLE)


    frag_with_anchors = efrag.GetMol()
    link_with_anchors = elink.GetMol()

    try:
        frag_anchor_smi = Chem.MolToSmiles(frag_with_anchors)
        link_anchor_smi = Chem.MolToSmiles(link_with_anchors)
    except RuntimeError:
        frag_anchor_smi = Chem.MolToSmiles(frag_with_anchors, canonical=False)
        link_anchor_smi = Chem.MolToSmiles(link_with_anchors, canonical=False)

    return frag_anchor_smi, link_anchor_smi
    
def logp(samples, pred_mols):
    logplist = []
    Clogp_list = []
    goal_ClogP =2.5

    for m in pred_mols:
        try:
            Chem.SanitizeMol(m)
            Chem.AssignStereochemistry(m, force=True, cleanIt=True)
            mol = Chem.AddHs(m)
            logp = Crippen.MolLogP(mol)
            logplist.append(logp)
        except:
            continue

    for logp in logplist:
        clogp = max(0.0, 1 - (1/6)*abs(logp - goal_ClogP))
        Clogp_list.append(clogp)
    rclogp = sum(Clogp_list) / len(Clogp_list)
    print(f'averaged rclogp value = {rclogp}')
    os.makedirs(samples,exist_ok=True)
    rclogp_value_path = os.path.join(samples, 'rclogp_value.txt')
    with open(rclogp_value_path, "w") as file:
        file.write(str(rclogp))

def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(os.getcwd(), name)
        # name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict
 
 
def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro
 
 
def calculateScore(m):
    if _fscores is None:
        readFragmentScores()
 
    # fragment score
    # AllChem.Compute2DCoords(m)
    # m = Chem.AddHs(m)
    # m = m.CalcImplicitValence()
    Chem.SanitizeMol(m)
    fp = rdMolDescriptors.GetMorganFingerprint(m,2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf
 
    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1
 
    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    # macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)
 
    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
 
    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5
 
    sascore = score1 + score2 + score3
 
    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
 
    return sascore

# def SA(samples, mols:list):
def SA(samples, mols):
    sa_list = []
    lg_sa_list = []
    readFragmentScores("fpscores")
    for m in mols:
        try:
            s = calculateScore(m)
            sa_list.append(s)
        except:
            continue
    for s in sa_list:
        lg_sa = math.log10(s)
        lg_sa_list.append(lg_sa)
    # print(f'sa_score = {sa_list} lg_sa_list = {lg_sa_list}')
    avg_sa = sum(lg_sa_list)/len(lg_sa_list)
    print(f'averaged sa value = {avg_sa}')
    # return avg_sa
    os.makedirs(samples,exist_ok=True)
    sa_value_path = os.path.join(samples, 'sa_value.txt')
    with open(sa_value_path, "w") as file:
        file.write(str(avg_sa))


def qed(samples, pred_mols):
    qed_list = []
    # goal_ClogP =2.5

    for m in pred_mols:
        try:
            Chem.SanitizeMol(m)
            Chem.AssignStereochemistry(m, force=True, cleanIt=True)
            mol = Chem.AddHs(m)
            qed = QED.qed(mol)
            qed_list.append(qed)
        except:
            continue

    avg_qed = sum(qed_list) / len(qed_list)
    print(f'averaged QED value = {avg_qed}')
    os.makedirs(samples,exist_ok=True)
    avg_qed_path = os.path.join(samples, 'qed_value.txt')
    with open(avg_qed_path, "w") as file:
        file.write(str(avg_qed))



def get_scoring_function(scoring_function, samples, pred_mols):
    scoring_function_dict = {'rclogp': logp, 'sa': SA, 'qed': qed}
    if scoring_function in scoring_function_dict:
        print(f'current selected function is {scoring_function}')
        selected_function = scoring_function_dict[scoring_function]
        selected_function(samples, pred_mols)
    else:
        raise ValueError(f"invalid {scoring_function} type")
    
def reformat(samples, true_smiles_path, scoring_function):
    true_smiles_path = os.path.join(true_smiles_path)

    reformat_smi = []

    true_smiles_table = pd.read_csv(true_smiles_path, sep=' ', names=['molecule', 'fragments'])
    idx2true_mol_smi = dict(enumerate(true_smiles_table.molecule.values))
    idx2true_frag_smi = dict(enumerate(true_smiles_table.fragments.values))

    pred_mols, pred_mols_smi, pred_link_smi, true_mols_smi, true_frag_smi = load_sampled_dataset(
        folder=samples,
        idx2true_mol_smi=idx2true_mol_smi,
        idx2true_frag_smi=idx2true_frag_smi,
    )
    # print(f"pred_mols_num = {len(pred_mols)}")
    # print(f'pred_mols_smi = {pred_mols_smi}')

    frag_anchor_smi = []
    link_anchor_smi = []

    for i in range(len(pred_mols)):
        try:
            frag_anchor, link_anchor = add_anchor(pred_mols_smi[i], true_frag_smi[i], pred_link_smi[i])
            # print(f'for loop: {frag_anchor, link_anchor}')
            # if frag_anchor_smi.count('*') == 2 and link_anchor_smi.count('*') ==2: 
            # print(f'list append: {frag_anchor, link_anchor}')
            frag_anchor_smi.append(frag_anchor)
            link_anchor_smi.append(link_anchor)
            # else:
            #     continue
            # frag_anchor_smi.append(frag_anchor)
            # link_anchor_smi.append(link_anchor)
        except Exception as e:
            print(e)
            frag_anchor_smi.append(None)
            link_anchor_smi.append(None)

    # print (f'anchor smi = {link_anchor_smi, frag_anchor_smi}')

    # logplist = []

    # for m in pred_mols:
    #     try:
    #         Chem.SanitizeMol(m)
    #         Chem.AssignStereochemistry(m, force=True, cleanIt=True)
    #         mol = Chem.AddHs(m)
    #         logp = Crippen.MolLogP(mol)
    #         logplist.append(logp)
    #     except:
    #         continue
    get_scoring_function(scoring_function, samples, pred_mols)
    # Clogp_list = []
    # goal_ClogP =2.5

    # for logp in logplist:
    #     clogp = max(0.0, 1 - (1/6)*abs(logp - goal_ClogP))
    #     Clogp_list.append(clogp)
    # rclogp = sum(Clogp_list) / len(Clogp_list)
    # print(rclogp)
    # os.makedirs(samples,exist_ok=True)
    # rclogp_value_path = os.path.join(samples, 'rclogp_value.txt')
    # with open(rclogp_value_path, "w") as file:
    #     file.write(str(rclogp))
    metrics_smi =[]
    metrics_path = os.path.join(samples, 'metrics.smi')  
    smi_path = os.path.join(samples, 'reformatsmi.smi')  
    sdf_path = os.path.join(samples, 'pred_mols.sdf')

    with Chem.SDWriter(open(sdf_path, 'w')) as writer:
            for i in range(len(link_anchor_smi)):
                reformat_smi.append(f'{pred_mols_smi[i]} {link_anchor_smi[i]} {frag_anchor_smi[i]}')
                metrics_smi.append(f'{true_frag_smi[i]} {true_mols_smi[i]} {pred_mols_smi[i]} {pred_link_smi[i]}')
                # print(f'reformat smi = 1. {pred_mols_smi[i]} 2. {link_anchor_smi[i]} 3. {frag_anchor_smi[i]}')
                name = pred_mols_smi[i]
                pred_mols[i].SetProp('_Name', str(name))
                writer.write(pred_mols[i]) 

    with open(smi_path, 'w') as f:
        for smi in reformat_smi:
            f.write(smi + "\n")    
            
    with open(metrics_path, 'w') as f:
        for smi in metrics_smi:
            f.write(smi + "\n")

    return reformat_smi 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', action='store', type=str, required=True) # input
    parser.add_argument('--out_dir', action='store', type=str, required=True) # output
    parser.add_argument('--true_smiles_path', action='store', type=str, required=True) # true molecular
    parser.add_argument('--template', action='store', type=str, required=True)
    parser.add_argument('--scoring_function', action='store', type=str, required=True) #scoring function type
    parser.add_argument('--n_samples', action='store', type=int, required=True)
    args = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--samples', action='store', type=str, required=False, default = "DiffLinker/sample1") # input
    # parser.add_argument('--out_dir', action='store', type=str, required=False, default = "DiffLinker/out_dir") # output
    # parser.add_argument('--true_smiles_path', action='store', type=str, required=False, default = "DiffLinker/datasets/zinc_final_test_smiles.smi") # true molecular
    # parser.add_argument('--template', action='store', type=str, required=False, default = "reformat_mol")
    # parser.add_argument('--n_samples', action='store', type=int, required=False, default = 1)
    # args = parser.parse_args()

    reformat_smi = reformat(
        samples=args.samples,
        true_smiles_path=args.true_smiles_path,
        scoring_function=args.scoring_function
    )

    processes = []
    for pid in range(args.n_samples):
        sdf_path = os.path.join(args.samples, f'pred_mols.sdf')
        # out_mol_path = os.path.join(args.out_dir, f'{args.template}_mol_{pid}.sdf')
        # out_frag_path = os.path.join(args.out_dir, f'{args.template}_frag_{pid}.sdf')
        # out_link_path = os.path.join(args.out_dir, f'{args.template}_link_{pid}.sdf')
        # out_table_path = os.path.join(args.out_dir, f'{args.template}_table_{pid}.csv')
        out_mol_path = os.path.join(args.out_dir, f'{args.template}_mol.sdf')
        out_frag_path = os.path.join(args.out_dir, f'{args.template}_frag.sdf')
        out_link_path = os.path.join(args.out_dir, f'{args.template}_link.sdf')
        out_table_path = os.path.join(args.out_dir, f'{args.template}_table.csv')
        kwargs = {
            'table_smi': reformat_smi,
            'sdf_path': sdf_path,
            'out_mol_path': out_mol_path,
            'out_frag_path': out_frag_path,
            'out_link_path': out_link_path,
            'out_table_path': out_table_path,
            'progress': pid == 0,
        }
        print(len(reformat_smi))
        # print(sdf_path)
        os.makedirs(args.out_dir, exist_ok=True)
        prep.run(reformat_smi, sdf_path, out_mol_path, out_frag_path, out_link_path, out_table_path)
        # process = mp.Process(target=prep.run, kwargs=kwargs)
        # process.start()
        # processes.append(process)

    # for pid in range(args.n_samples):
    # for pid in range(3):
    #     processes[pid].join()
