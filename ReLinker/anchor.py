from rdkit import Chem
# CCC1=CSC(N2CCCC2N(CC)CC2=CC(C)=CC(C)=C2)=N1 C1CCN(c2nc(CC)cs2)C1.Cc1cccc(C)c1 CCNC
frag = Chem.MolFromSmiles("C1CCN(c2nc(CC)cs2)C1.Cc1cccc(C)c1")
link = Chem.MolFromSmiles("CCNC")
mol = Chem.MolFromSmiles("CCC1=CSC(N2CCCC2N(CC)CC2=CC(C)=CC(C)=C2)=N1")

bonds = list(mol.GetBonds())
matches1 = mol.GetSubstructMatches(frag)
match1 = []
match2 = []

matches2 = mol.GetSubstructMatches(link)
print(f"matches= {matches1, matches2}")
for match_1 in matches1:
    for match_2 in matches2:
        if not set(match_1).intersection(set(match_2)):
            match1.extend(match_1)
            match2.extend(match_2)

print(f"match1= {match1}")
print(f"match2= {match2}")

def bond_sort_key(bond):
    return bond.GetIdx()

efrag = Chem.EditableMol(frag)
elink = Chem.EditableMol(link)

for bond in sorted(bonds, key=bond_sort_key, reverse=True):
    begin_atom = bond.GetBeginAtom() 
    end_atom = bond.GetEndAtom()
    anchor = Chem.Atom('*')

    if begin_atom.GetIdx() in match1 and end_atom.GetIdx() in match2:

        anchor11_idx = begin_atom.GetIdx()
        anchor11 = match1.index(anchor11_idx)
        print(f"atom11_idx= {anchor11_idx}")
        idx11 = efrag.AddAtom(anchor)
        efrag.AddBond(anchor11, idx11, Chem.rdchem.BondType.SINGLE)
    
        anchor12_idx = end_atom.GetIdx()
        anchor12 = match2.index(anchor12_idx)
        print(f"atom12_idx= {anchor12_idx}")
        idx12 = elink.AddAtom(anchor)
        elink.AddBond(anchor12, idx12, Chem.rdchem.BondType.SINGLE)
        
    if begin_atom.GetIdx() in match2 and end_atom.GetIdx() in match1:

        anchor21_idx = begin_atom.GetIdx()
        anchor21 = match2.index(anchor21_idx)
        print(f"atom21_idx= {anchor21_idx}")
        idx21 = elink.AddAtom(anchor)
        elink.AddBond(anchor21, idx21, Chem.rdchem.BondType.SINGLE)
    
        anchor22_idx = end_atom.GetIdx()
        anchor22 = match1.index(anchor22_idx)
        print(f"atom22_idx= {anchor22_idx}")
        idx22 = efrag.AddAtom(anchor)
        efrag.AddBond(anchor22, idx22, Chem.rdchem.BondType.SINGLE)


frag_with_anchors = efrag.GetMol()
link_with_anchors = elink.GetMol()

try:
    fraganchorsmi = Chem.MolToSmiles(frag_with_anchors)
    linkanchorsmi = Chem.MolToSmiles(link_with_anchors)
except RuntimeError:
    fraganchorsmi = Chem.MolToSmiles(frag_with_anchors, canonical=False)
    linkanchorsmi = Chem.MolToSmiles(link_with_anchors, canonical=False)


print("Frag SMILES with anchors:", fraganchorsmi)
print("Link SMILES with anchors:", linkanchorsmi)

