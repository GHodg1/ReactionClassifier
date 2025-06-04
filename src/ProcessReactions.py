
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import SaltRemover
from rxnmapper import RXNMapper
import argparse
from tqdm import tqdm


def process_all_reactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all reactions in the DataFrame and return a new DataFrame with new single step reactions.
    """
    processed_rows = []
    # process rows and add a tqdm progress bar
    print("Generating single product atom-mapped reactions.")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reactions"):
        try:
            reaction_processor = ProcessReaction(row)
            reaction_processor.develop_single_step_reactions()
            processed_rows.extend(reaction_processor.single_step_reactions)
        except Exception as e:
            print(f"Error processing row {_}: {e}")

    return pd.DataFrame(processed_rows)


class ProcessReaction:
    """
    Extracts reactions based on reaction class.

    First we process the OPRD reactions by stripping salts and separating reactions into single product reactions.

    Then we perform atom-atom mapping with rxnmapper - discarding salt formation reactions.

    Finally, we extract the reaction class based on the reaction templates defined above and via a SMARTS checker for heterocycle formation.
    """

    def __init__(self, row: pd.Series):
        self.row = row
        self.reaction_smiles = row["Reaction"].strip()
        self.reaction_class = None
        self.single_step_reactions = []
        

    def map_reaction(self,reaction_smiles):
        """
        Map reaction SMILES using RXNMapper and add the mapped reaction to the reaction row entry.
        """
        # Check if the reaction SMILES is valid
        reaction = rdChemReactions.ReactionFromSmarts(reaction_smiles)
        if not reaction:
            raise ValueError("Invalid reaction SMILES")
        # Map the reaction using RXNMapper
        rxn_mapper = RXNMapper()
        mapped_reaction = rxn_mapper.get_attention_guided_atom_maps([reaction_smiles])
        if not mapped_reaction:
            raise ValueError("Failed to map the reaction")
        # Add the mapped reaction to the data dictionary
        mapped_reaction, confidence = mapped_reaction[0].get("mapped_rxn"), mapped_reaction[0].get("confidence")
        return mapped_reaction, confidence

    def develop_single_step_reactions(self):
        # Create a reaction object from the SMILES string
        reaction = rdChemReactions.ReactionFromSmarts(self.reaction_smiles)
        if not reaction:
            raise ValueError("Invalid reaction SMILES")
        # Get the reactants and products from the reaction SMILES
        reactants, products = self.reaction_smiles.split(">>")
        if not reactants or not products:
            raise ValueError("Reaction SMILES improperly constructed")
        # Strip Salts from the reaction SMILES
        defnData = """[H+]
            [Cl,Br,I]
            [Li,Na,K,Ca,Mg,Ba]
            [O,N]
            [N](=O)(O)O
            [P](=O)(O)(O)O
            [P](F)(F)(F)(F)(F)F
            [S](=O)(=O)(O)O
            [CH3][S](=O)(=O)(O)
            c1cc([CH3])ccc1[S](=O)(=O)(O)
            [CH3]C(=O)O
            FC(F)(F)C(=O)O
            OC(=O)C=CC(=O)O
            OC(=O)C(=O)O
            OC(=O)C(O)C(O)C(=O)O
            C1CCCCC1[NH]C1CCCCC1
            O=C([C@H](OC(C1=CC=CC=C1)=O)[C@@H](OC(C2=CC=CC=C2)=O)C(O)=O)[O-]
            [O-]C(C(O)=O)=O
            O=C(O)CC(C([O-])=O)(O)CC(O)=O
            CC1(C2CCC1(C(C2)=O)CS(=O)([O-])=O)C
            O=C([O-])[C@H](O)[C@@H](O)C(O)=O
            CC[NH+](CC)CC
            CS(=O)(O)=O
            [H][C@]12CC[C@](C2(C)C)(CS(O)(=O)=O)C(C1)=O
            CC2([C@@H]3CC[C@]2(C(C3)=O)CS(O)(=O)=O)C
            CC2=CC=C(S(=O)(O)=O)C=C2
            CC1(C)[C@@H]2CC[C@@]1(CS(O)(=O)=O)C(C2)=O
            FC(C([O-])=O)(F)F
            FB(F)F
            CCCC[N](CCCC)(CCCC)CCCC
            [N-]=[N+]=[N-]
            CC[N+](CC)(CC)CC
            O=C([O-])O
            CCCCCCCCCCCCCCCC[N+](C)(C)C"""
        remover = SaltRemover.SaltRemover(defnData=defnData,defnFormat="smarts")
        try:
            mol = Chem.MolFromSmiles(products.strip())
            if mol is None:
                raise ValueError("Invalid product SMILES before salt stripping")
            products_mol = remover.StripMol(mol)
            if products_mol is None or products_mol.GetNumAtoms() == 0:
                raise ValueError("No atoms left in product after salt stripping.")
            products = Chem.MolToSmiles(products_mol)
            #print("Products after salt stripping:", products)
            # remove starting and trailing "." from the products string
            products_list = []
            for i in products.split("."):
                i = i.strip()
                if not i or i == "":
                    continue
                mol = Chem.MolFromSmiles(i)
                if mol is None:
                    raise ValueError(f"Invalid SMILES encountered: '{i}'")
                products_list.append(mol)
        except Exception as e:
            raise ValueError(f"Invalid product SMILES: {e}")
        # Check if there are multiple products
        if len(products_list) > 1:
            # Check if the length of the yields list is equal to the number of products
            # there are two feasible sources of yield, columns: Yield and Yield (numerical)
            # We need to split at ";" if there are multiple yields, and assign None if there is no yield
            if "Yield" in self.row and isinstance(self.row["Yield"], str):
                yields = self.row["Yield"].split("; ")
                if len(yields) != len(products_list):
                    yields.extend([None] * (len(products_list) - len(yields)))
            else:
                yields = [None] * len(products_list)

            if "Yield (numerical)" in self.row and isinstance(self.row["Yield (numerical)"], str):
                yields_num = self.row["Yield (numerical)"].split("; ")
                if len(yields_num) != len(products_list):
                    yields_num.extend([None] * (len(products_list) - len(yields_num)))
            else:
                yields_num = [None] * len(products_list)
            # Now we can create new entries for single step reactions
            for i, product in enumerate(products_list):
                new_row = self.row.copy()
                product_smiles = Chem.MolToSmiles(product)
                if not product_smiles or product_smiles.strip() == "":
                    raise ValueError("No product SMILES found after salt stripping.")
                new_row["Reaction"] = f"{reactants}>>{product_smiles}"
                new_row["Yield"] = yields[i]
                new_row["Yield (numerical)"] = yields_num[i]
                mapped_reaction, confidence = self.map_reaction(new_row["Reaction"])
                new_row["Mapped Reaction"] = mapped_reaction
                new_row["Confidence"] = confidence
                self.single_step_reactions.append(new_row)
        else:
            # If only one product, return the salt stripped reaction
            new_row = self.row.copy()
            product_smiles = Chem.MolToSmiles(products_list[0])
            if not product_smiles or product_smiles.strip() == "":
                raise ValueError("No product SMILES found after salt stripping.")
            new_row["Reaction"] = f"{reactants}>>{product_smiles}"
            mapped_reaction, confidence = self.map_reaction(new_row["Reaction"])
            new_row["Mapped Reaction"] = mapped_reaction
            new_row["Confidence"] = confidence
            self.single_step_reactions.append(new_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    # limit to first 500 rows for testing
    #df = df.head(500) 
    if "Reaction" not in df.columns:
        raise ValueError("Input file must contain 'Reaction' column containing reaction SMILES.")
    processed_df = process_all_reactions(df)
    processed_df.to_csv(args.output, index=False)
    print(len(df), "one step reactions loaded from input file.")
    print(len(processed_df), "single step reactions processed and saved to output file.")


