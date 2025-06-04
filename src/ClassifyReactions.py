import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from rdkit.Chem import rdChemReactions as Reactions
from tqdm import tqdm
import itertools
import os
import argparse

# Define a dictionary of the 20 most common MedChem reactions found in the 2015 study by Brown and coworkers (J. Med. Chem. 2016, 59, 10, 4443–4458)
# Heterocycle formation reactions are not included in this list, as they are handled separately.
REACTION_TEMPLATES = {
    "Buchwald-Hartwig Coupling":{ 
        "Template":[
        "[c:1][Cl,Br,I].[n,N;H1,H2:2]>>[*:1]-[*:2]", # Aryl halide with amine
        "[C;$([CH2]([c])[NX4;+1]):1].[n,N;H1,H2:2]>>[*:1]-[*:2]" # Special case with benzyl ammonium salt
    ],
        "Reagents_include": "Palladium|Pd",
        "Reagents_exclude": None,
    },
    "Aromatic Halogenation":{ 
        "Template":[
            "[cH:1]>>[c:1][Cl,Br,I]", # Aromatic halogenation
            "[cH:1].[Cl,Br,I;-1:2]>>[*:1][*:2]",
            "[cH:1].[Cl,Br,I;+0:2]>>[*:1][*:2]"
        ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Wittig Olefination":{ 
        "Template":[
            "[CX3;H1,H0:1]=[O].[C:2]-[P](=O)(O)O>>[*:1]=[*:2]", # Wittig-Horner olefination
            "[CX3;H1,H0:1]=[O].[C:2]-[PX4;+1]>>[*:1]=[*:2]", # Wittig olefination with phosphonium salt
            "[CX3;H1,H0:1]=[O].[C:2]=[PX4;+0]>>[*:1]=[*:2]", # Wittig with phosphonium ylide
            "[CX3;H1,H0:1]=[O].[C:2]-[PX5;+0]>>[*:1]=[*:2]", # Wittig with pentavalent phosphorous
        ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Sonogashira Coupling":{ 
        "Template":[
            "[c:1][Cl,Br,I].[CX2H1:3]#[CX2:4]>>[c:1]-[C:3]#[C:4]", # Aryl halide with alkyne
        ],
        "Reagents_include": "Cuprous|Cu|Copper|Palladium|Pd",
        "Reagents_exclude": None,
    },
    "Alcohol Oxidation": {
        "Template":[
            "[C:1][OH]>>[C:1]=[O]", # Alcohol to aldehyde
            "[C:1][OH]>>[C:1](=[O])[O]", # Alcohol to carboxylic acid
        ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Sandmeyer Reaction": {
        "Template":[
            "[c:1][NX3H2]>>[c:1][Cl,Br,I]", # Aniline to aryl halide via diazonium salt
            "[c:1][NX3H2]>>[c:1][C](F)(F)F", # Aniline to CF3
            "[c:1][NX3H2]>>[c:1][CH](F)F", # Aniline to CF2H
            "[c:1][NX3H2]>>[c:1][C](=O)O", # Aniline to carboxylic acid
            "[c:1][NX3H2]>>[c:1][C]#N", # Aniline to cyano
            "[c:1][NX3H2]>>[c:1][BX3]", # Aniline to boronate
            "[c:1][NX3H2]>>[c:1][O]", # Aniline to Aryl ether/Phenol
            "[c:1][NX3H2]>>[c:1][S]", # Aniline to Aryl thioether/ Sulfone/Sulfoxide
            "[c:1][NX3H2]>>[c:1][#1]", # Aniline to c-H
            "[c:1][NX3H2]>>[c:1][#6]", # Aniline to c-C,c
        ],
        "Reagents_include": "sodium nitrite",
        "Reagents_exclude": None,
    },
    "Nitro Reduction": {
        "Template": [
         # Nitro group reduction to amine
        "[NX3+:1](=O)[O-]>>[NH2+0:1]",
        "[NX3:1](=O)=O>>[NH2+0:1]",
    ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Sulfonamide Formation": { 
        "Template":[
        "[NX3;H2,H1:1].[S:2](=[O:3])(=[O:4])[Cl,Br,F]>>[S:2](=[O:3])(=[O:4])[NX3;H0:1]", # Sulfonamide formation from amine and sulfonyl halide
        "[NX3;H2,H1:1].[S:2](=[O:3])(=[O:4])[Cl,Br,F]>>[S:2](=[O:3])(=[O:4])[NX3;H1:1]",
    ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Phenol Alkylation": { 
        "Template": [
        "[c:1][OH:2].[C,c:3]>>[c:1][O:2]-[*:3]", # Phenol alkylation
    ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Debenzylation":{
        "Template": [
        "[OX2:1]Cc1ccccc1>>[OX2H1:1]",
    ],
        "Reagents_include": "Palladium|Pd",
        "Reagents_exclude": None,
    },
    "Reductive Amination": {
        "Template":[
        "[C:1]=[O].[NX3H2:2]>>[*:1]-[*:2]", # Reductive amination with amine
        "[C:1]=[O]>>[C:1]-[NX3;H2,H1]", # Reductive amination with amine
    ],
        "Reagents_include": "Borohydride|Sodium cyanoborohydride|NaCNBH3|Sodium triacetoxyborohydride|NaBH(OAc)3",
        "Reagents_exclude": None,
    },
    "SNAr":{
        "Template": [
        "F-[c;H0;D3;+0:1](:[c:2]):[c:3].[#7;a:4]:[nH;D2;+0:5]:[#7;a:6]>>[#7;a:4]:[n;H0;D3;+0:5](:[#7;a:6])-[c;H0;D3;+0:1](:[c:2]):[c:3]",
        "[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]-[F,Cl].[c,n]:[nH;D2;+0:3]:[c,n]>>[*:3]-[*:1]", # Aromatic N
        "[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]-[F,Cl]>>[N]-[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]", # N nucleophile
        "[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]-[F,Cl]>>[n]-[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]", # N nucleophile
        "[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]-[F,Cl].[N,n:2]>>[*:2]-[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]",# N nucleophile
        "[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]-[F,Cl]>>[S]-[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]", # S nucleophile
        "[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]-[F,Cl].[S:2]>>[S:2]-[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]",# S nucleophile
        "[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]-[F,Cl]>>[O]-[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]", # O nucleophile
        "[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]-[F,Cl].[O:2]>>[O:2]-[c;$(c([F,Cl])a([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([F,Cl])aaa([$([NX3](=O)=O),$([NX3+](=O)[O-]),$(C=O),$(C#N),$(S=O)])),$(c([Cl,F])n),$(c([Cl,F])an),$(c([Cl,F])aan):1]", # O nucleophile   
    ],
        "Reagents_include": None,
        "Reagents_exclude": "Copper|Cu|Palladium|Pd|Nickel|Ni|Lithium|Li|Zinc|Zn|Magnesium|Mg|Iron|Fe|Tin|Sn|Boron|B",
    },
    "Boc Protection": {
        "Template":[
        "[N:1]>>[N:1][C](=[O])[O][C]([C])([C])[C]", # Boc protection of amine
    ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Boc Deprotection": {
        "Template":[
        "[N:1][C](=[O])[O][C]([C])([C])[C]>>[N:1]", # Boc deprotection of amine
    ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Electrophile with Amine": {
        "Template":[
        "[NX3;H2,H1:1].[C:2][Cl,Br,I]>>[NX3:1][C:2]", # Amine SN2 with alkyl halide
        "[NX3;H2,H1:1].[C:2][O;$(OS(=O)(=O)C(F)(F)F)]>>[NX3:1][C:2]", # Amine SN2 with alkyl triflate
    ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Amide Bond Formation": {
        "Template":[
        "[NH2:1][*:2][*:3][C:4](=[O:5])O>>[N:1]1[*:2][*:3][C:4](=[O:5])1", # Cyclic amide formation
        "[N;$([NX3;H2,H1]);!$([N]-[C]=[O,N]):1].[C;$([C]([OH])(=[O])):2](=[O:3])[O]>>[NX3:1]-[C:2](=[O:3])" # Amide formation from carboxylic acid and amine
    ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Esterification": {
        "Template":[
        "[C:1](=[O:2])[O].[C:4][OH]>>[C:1](=[O:2])[O][C:4]", # Esterification with alcohol
        "[C:1](=[O:2])[O]>>[C:1](=[O:2])[O][C]",
    ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Ester Hydrolysis":{
        "Template":[
        "[C:1](=[O:2])[O][C:4]>>[C:1](=[O:2])[O].[C:4][OH]", # Ester hydrolysis
        "[C:1](=[O:2])[O][C]>>[C:1](=[O:2])[O]", # Ester hydrolysis leaving acid
        "[C:1](=[O])[O][C]>>[C:1][OH]", # Ester hydrolysis leaving alcohol
    ],
        "Reagents_include": None,
        "Reagents_exclude": None,
    },
    "Suzuki Miyaura Coupling": {
        "Template":[
        "[c:1][Cl,Br,I].[c:3][BX3]>>[c:1]-[c:3]", # Suzuki coupling with aryl halide
        "[c:1][O;$(OS(=O)(=O)C(F)(F)F)].[c:3][BX3]>>[c:1][c:3]", # Suzuki coupling with triflate
        "[C:1]=[C:2][Cl,Br,I].[c:4][BX3]>>[C:1][C:2][c:4]", # Suzuki coupling with alkenyl halide
        "[C:1]=[C:2][O;$(OS(=O)(=O)C(F)(F)F)].[c:4][BX3]>>[C:1][C:2][c:4]", # Suzuki coupling with alkenyl halide and triflate
        "[c:1][Cl,Br,I].[C:3][BX3]>>[c:1]-[C:3]",
        "[c:1][O;$(OS(=O)(=O)C(F)(F)F)].[C:3][BX3]>>[c:1][C:3]",
        "[c:1][Cl,Br,I].[C:3][BX3]>>[c:1]-[C:3]",
        "[c:1][O;$(OS(=O)(=O)C(F)(F)F)].[C:3][BX3]>>[c:1][C:3]",
        "B(O)(O)([#6:1]).Cl[#6:2]>>[*:1][*:2]"
    ],
        "Reagents_include": "Pd|Palladium",
        "Reagents_exclude": None,
    },
}
   

def process_all_reactions(df: pd.DataFrame, templates: dict, smarts: pd.DataFrame = None):
    """
    Process all reactions in the DataFrame and classify them.
    """
    processed_rows = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Classifying reactions"):
        try:
            classifier = ClassifyReactions(row, templates, smarts=smarts)
            updated_row = classifier.classify()  # Get the updated row
            processed_rows.append(updated_row)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    return pd.DataFrame(processed_rows)

def init_rxn_dict(templates_dict):
    """
    Add a 'ReactionObjects' key to each reaction entry in the provided dictionary,
    containing the list of compiled RDKit reaction objects.
    """
    for template_name, template_info in templates_dict.items():
        rxns = []
        template_list = template_info.get("Template", [])
        for template in template_list:
            try:
                rxn = Reactions.ReactionFromSmarts(template)
                print(f"Initialized reaction template: {template_name} with SMARTS: {template}")
                rxns.append(rxn)
            except Exception as e:
                print(f"Error processing template {template_name}, {template}: {e}")
        template_info["ReactionObjects"] = rxns
    return templates_dict

class ClassifyReactions:

    def __init__(self, row: pd.Series, templates: dict, smarts: pd.DataFrame = None):
        """
        Initialize the ClassifyReactions object with a DataFrame row and reaction templates.

        Args:
            row (pd.Series): A row from the DataFrame containing reaction data.
            templates (dict): A dictionary of reaction templates.
        """
        self.row = row
        self.reaction_smiles = row["Mapped Reaction"]
        self.remove_unmapped_reactants()  # Clean the reaction SMILES
        self.templates = templates
        self.smarts = smarts


    def get_FG_array(self,SMARTS, smiles)->np.array:
        # Get SMARTS patterns
        SMARTS_codes = SMARTS["SMARTS"].tolist()
        # where the FG matches will be recorded
        FG_list = []
        for smile in smiles:  # for every molecule
            FG = []  # each new row of output
            # Create molecule object from SMILES
            mol = Chem.AddHs(Chem.MolFromSmiles(smile))
            # match all SMARTS to SMILES, append "0" if no match, otherwise append 1
            for smarts in SMARTS_codes:
                patt = Chem.MolFromSmarts(smarts)
                if mol.HasSubstructMatch(patt):
                    FG.append(len(mol.GetSubstructMatches(patt)))
                else:
                    FG.append(0)
            # add to master list
            FG_list.append(FG)
        # make a pandas dataframe
        FG_array = np.array(FG_list)
        return FG_array

    def check_heterocyle(self, smarts, reactants: list[str], products: list[str]):
        """
        Check if reaction produces a heterocycle - difficult to do with reaction SMARTS - so we use FG matching.
        """
        # Check if number of rings in products is greater than in reactants
        num_rings_reactants = sum(CalcNumRings(Chem.MolFromSmiles(r)) for r in reactants)
        num_rings_products = sum(CalcNumRings(Chem.MolFromSmiles(p)) for p in products)
        if num_rings_products > num_rings_reactants:
            # If any product has a FG match that the reactants don't, it's a heterocycle
            arrays = []
            for mols in [reactants, products]:
                FG_array = self.get_FG_array(smarts, mols)
                if FG_array.size == 0:
                    return None
                FG_counts = np.sum(FG_array, axis=0)
                arrays.append(FG_counts)
            product_counts, reactant_counts = arrays
            if len(product_counts) != len(reactant_counts):
                raise ValueError("Product and reactant FG counts must have the same length.")
            product_counts = pd.Series(product_counts, index=smarts["Name"].tolist())
            reactant_counts = pd.Series(reactant_counts, index=smarts["Name"].tolist())
            difference = reactant_counts - product_counts
            difference[difference < 0] = 0
            # Find non-zero indexes
            non_zero_indexes = difference[difference > 0].index.tolist()
            if not non_zero_indexes:
                return None
            heterocycle = smarts[smarts["Name"].isin(non_zero_indexes)]
            # join the names of the heterocycles into a single string
            heterocycle_names = ", ".join(heterocycle["Name"].tolist())
            return heterocycle_names
        return None

    def get_r_p(self, reaction_smiles: str)->list:
        """
        Split reactants and products from a reaction SMILES string.
        """
        reactants = reaction_smiles.split(">>")[0]
        reactant_smiles = reactants.split(".")
        reactant_mols = [Chem.MolFromSmiles(i) for i in reactant_smiles]
        products = reaction_smiles.split(">>")[1]
        products_smiles = products.split(".")
        products_mols = [Chem.MolFromSmiles(i) for i in products_smiles]
        return reactant_mols, products_mols

    def mol_not_mapped(self, mol):
        return all(atom.GetAtomMapNum() == 0 for atom in mol.GetAtoms())

    def remove_unmapped_reactants(self):
        """
        Remove unmapped molecules from the reaction SMILES
        """
        reactant_mols, product_mols = self.get_r_p(self.reaction_smiles)
        # loop over molecules, remove those without map numbers
        reactant_mols = [mol for mol in reactant_mols if not self.mol_not_mapped(mol)]
        # Reconstruct the reaction SMILES with only mapped reactants
        if reactant_mols:
            reactant_smiles = ".".join(Chem.MolToSmiles(mol) for mol in reactant_mols)
            product_smiles = ".".join(Chem.MolToSmiles(mol) for mol in product_mols)
            self.reaction_smiles = f"{reactant_smiles}>>{product_smiles}"
        
        else:
            raise ValueError("No mapped reactants found in the reaction SMILES.")

    # Function to check if proposed products match any actual products completely
    def products_match(self, proposed_products, product_mols, rxn_name):
        for product_tuple in proposed_products:
            for proposed_product in product_tuple:
                for actual_product in product_mols:
                    # Get all substructure matches
                    matches = actual_product.GetSubstructMatches(proposed_product)
                    # Check if any match covers the entire proposed product
                    if any(len(match) == proposed_product.GetNumAtoms() for match in matches):
                        if rxn_name =="SNAr": # SNAr reactions often don't include the amine reactant in rxn SMILES - so we needed a more general template
                            return True
                        else:
                            # Check the tanimoto similarity between the proposed and actual products
                            proposed_fp = Chem.RDKFingerprint(proposed_product)
                            actual_fp = Chem.RDKFingerprint(actual_product)
                            tanimoto_sim = Chem.DataStructs.TanimotoSimilarity(proposed_fp, actual_fp)
                            # If tanimoto similarity is 1, consider it a match
                            if tanimoto_sim == 1.0:
                                return True
        return False

    def classify(self):
        # Initialize with None if not salt metathesis
        self.row["Reaction Class"] = None

        if self.row.get("Reaction Template") == "Salt metathesis":
            self.row["Reaction Class"] = "Salt metathesis"
            return self.row

        try:
            reactant_mols, product_mols = self.get_r_p(self.reaction_smiles)
            if not reactant_mols or not product_mols:
                raise ValueError("Invalid reaction SMILES provided.")
        except Exception as e:
            print(f"Error parsing SMILES for row: {self.reaction_smiles}, Error: {e}")
            return self.row

        for template_name, template_info in self.templates.items():
            rxn_list = template_info.get("ReactionObjects", [])
            rgts_include = template_info.get("Reagents_include", None)
            rgts_exclude = template_info.get("Reagents_exclude", None)
            for template in rxn_list:
                try:
                    template_match = False
                    if template.GetNumReactantTemplates() == 1:
                        for reactant in reactant_mols:
                            proposed_products = template.RunReactants((reactant,))
                            if self.products_match(proposed_products, product_mols, template_name):
                                template_match = True                              
                    else:
                        for reactant_combination in itertools.combinations(reactant_mols, template.GetNumReactantTemplates()):
                            proposed_products = template.RunReactants(reactant_combination)
                            if self.products_match(proposed_products, product_mols, template_name):
                                template_match = True
                    if template_match:
                        if rgts_include:
                            if self.row["Reagent"] and re.search(rgts_include, str(self.row["Reagent"]), re.IGNORECASE):
                                template_match = True
                        if rgts_exclude:
                            if self.row["Reagent"] and re.search(rgts_exclude, str(self.row["Reagent"]), re.IGNORECASE):
                                template_match = False
                    if template_match:
                        self.row["Reaction Class"] = template_name
                        return self.row
                except Exception as e:
                    print(f"Error processing template {template_name}: {e}")
        # If no match found with SMARTS templates, check for heterocycles
        if self.row["Reaction Class"] is None:
            if self.smarts is not None:
                # Check there arent any missing reactant atom map numbers in the products
                reactant_atom_map_nums = [atom.GetAtomMapNum() for mol in reactant_mols for atom in mol.GetAtoms()]
                product_atom_map_nums = [atom.GetAtomMapNum() for mol in product_mols for atom in mol.GetAtoms()]
                # If all product atom map numbers are in the reactant atom map numbers, we can check for heterocycles
                if all(atom_map in reactant_atom_map_nums for atom_map in product_atom_map_nums):
                    reactant_smiles = [Chem.MolToSmiles(mol) for mol in reactant_mols]
                    product_smiles = [Chem.MolToSmiles(mol) for mol in product_mols]
                    heterocycle_names = self.check_heterocyle(self.smarts, reactant_smiles, product_smiles)
                    if heterocycle_names:
                        self.row["Reaction Class"] = f"Heterocycle formation: {heterocycle_names}"
                        return self.row

        return self.row



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input)

    # Import smarts of heterocycles
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    smarts = pd.read_csv(os.path.join(cwd, "data/heterocycle_SMARTS.csv"))
    # Check that all SMARTS are valid
    for smart in smarts["SMARTS"].tolist():
        if Chem.MolFromSmarts(smart) is None:
            print(f"Invalid SMARTS: {smart}")

    # Read the input CSV file into a DataFrame
    df = pd.read_csv(args.input)
    # Initialize the reaction templates
    rxn_templates = init_rxn_dict(REACTION_TEMPLATES)
    # Process the DataFrame to classify reactions
    classified_df = process_all_reactions(df, rxn_templates, smarts=smarts)
    # Save the classified DataFrame to a new CSV file
    classified_df.to_csv(args.output, index=False)
    print(f"Classified reactions saved to {args.output}")

