"""
This script generates Reaction Templates for the post-processed reaction data.

We take advantage of the ReactionUtils package provided by MolecularAI to generate SMARTS reaction templates for reactions.

These templates are stored in the JSON data for each reaction in the post-processed data, the tag is ["Reaction Template"].

This script has to be run with the specific rxn-utils-env environment.
"""



import pandas as pd
from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.template import ReactionTemplate
from tqdm import tqdm
import argparse

def process_all_reactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all reactions in the DataFrame and return a new DataFrame with reaction templates.
    """
    processed_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reactions"):
        try:
            template_generator = AddTemplates(row)
            processed_row = template_generator.generate_template(row)
            processed_rows.append(processed_row)
        except Exception as e:
            print(f"Error processing row {_}: {e}")

    return pd.DataFrame(processed_rows)


class AddTemplates:

    def __init__(self, row: pd.Series):
        self.row = row
    
    def generate_template(self, row: pd.Series)-> pd.Series:
        mapped_reaction = row["Mapped Reaction"]
        if mapped_reaction:
            try:
                # Create a ChemicalReaction object from the mapped reaction
                reaction = ChemicalReaction(mapped_reaction)
                # Generate the SMARTS template at radius of 1 atom
                forward_template = reaction.generate_reaction_template(radius=1)[0].smarts
                # Generate the difference MFP2
                template = ReactionTemplate(forward_template)
                mfp2_diff = template.fingerprint_vector(radius=2, nbits=1024, use_chirality=True)
                # Add the template to the reaction row entry
                row["Reaction Template"] = str(forward_template)
                row["MFP2_difference"] = mfp2_diff.tolist()
            except Exception as e:
                if "no change in reaction" in str(e):
                    row["Reaction Template"] = "Salt metathesis"
                    row["MFP2_difference"] = None
                else:
                    row["Reaction Template"] = None
                    row["MFP2_difference"] = None
        else:
            row["Reaction Template"] = None
            row["MFP2_difference"] = None
        return row
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate Reaction templates and find Salt-metathesis reactions.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output CSV file")
    args = parser.parse_args()

    print("Generating reaction templates...")

    # Define the input and output file paths
    input_file_path = args.input
    output_file_path = args.output

    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_file_path)
    # Process the DataFrame to add reaction templates
    processed_df = process_all_reactions(df)
    # Save the processed DataFrame to a new CSV file
    processed_df.to_csv(output_file_path, index=False)
    print(f"Processed reactions with templates saved to {output_file_path}")















