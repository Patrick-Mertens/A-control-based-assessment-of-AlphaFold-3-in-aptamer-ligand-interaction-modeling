#!/usr/bin/env python3
"""
generate_af3_aptamer_json_cocaine_study_v2.py

Alphafold3 Aptamer-Ligand JSON input generator (buffer-aware, enhanced).

What this script does:
- Read aptamer from a CSV (headers: table, target, aptamer_id, sequence)
- Generates one AlphaFold3 JSON input per aptamer for a chosen target ligand
- Optionally approximates an experimental buffer by adding discrete copies of small molecules (ions + Tris) using CCD codes and SMILES.



IMPORTANT: Concentration-to-Discrete-Copy Conversion Limitations
==================================================================================================
AlphaFold3 inputs do not accept molar concentrations. This script converts mM values into *RELATIVE COUNTS* under a
configurable cap. This is an approximation with known limitations:

1. RATIO DISTORTION: When scaling from mM to discrete copies, the relative proportions change significantly. For example, the cocaine buffer:
    Original (mM) for cocaine selection buffer:         Tris-HCl:NaCl:KCl:MgCl₂ = 20:140:4:5   -> ratios 4:28:0.8:1
    MScaled to Mg (n=25) for cocaine selection buffer:  Tris-HCl:NaCl:KCl:MgCl₂ = 3:20:1:1     -> ratios 3:20:1:1
    OGScaled for Na (n=11) for cocaine selection buffer:Tris-HCl:NaCl:KCl:MgCl₂ = 1:8:1:1      -> ratios 1:8:1:1

    K⁺ and Mg²⁺ becine iverreoresebted relative to Na⁺. This may affect predictions if ionic competition is relevant.

2. STRUCTURAL vs THERMODYNAMIC: AF3 uses ions for structural coordination, not thermodynamic ionic strenght calculations. The discrete copies represent potential coordination sites, not solution concentrations.

3. RECOMMENDED APPROACH: Use an torough experimental setup, to gain information through inference of each setting/enviroment and such.

Buffer presets:
- cocaine: 20 mM Tris-HCl, 140 mM NaCl, 4 mM KCl, 5 mM MgCl₂


#TODO: Need to update this
Example usage:
    #Default optimized buffer (4 Mg²⁺ + minimal K/Na)
    python generate_af3_aptamer_json_cocaine_study_v2.py cocaine cocaine \\
    --csv aptamer_sequences.csv --output-dir af3_inputs --seeds 1-5

    #Full proportional buffer scaling
    python generate_af3_aptamer_json_cocaine_study_v2.py cocaine cocaine \\
    --buffer cocaine --buffer-mode=proportional --seeds 1-5

    #Dry run to preview allocation without writing files
    python generate_af3_aptamer_json_cocaine_study_v2.py cocaine cocaine \\
    --buffer cocaine --dry-run

    #Override Mg count while keeping charge-balanced Cl
    python generate_af3_aptamer_json_cocaine_study_v2.py cocaine cocaine \\
    --buffer cocaine --mg 4 --seeds 1-5

    #Mg-only mode (recommended based on prior optimization AND it allows to run with only Aptamer and Target)
    python generate_af3_aptamer_json_cocaine_study_v2.py cocaine cocaine \\
    --buffer none --mg 4 --seeds 1-5

#TODO: Need to update this
NOTES:
PRM_N0: While this script was genrated with Claude, I am currently typing it over; improving, modyfing, and checking while I am writing.

PRM_N1: I modified the proportional buffer, to Mproportional buffer and OGproportional buffer, the OGproportional buffer was used for the conference paper; however, while being relatively nice for Na-ions, I think the ratio could be more proportional

PRM_N2: TODO0: I need to check if SMILES cause a difference in results when used as input

PRM_N3: TODO1: I need to add al compounds
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import difflib
import os


# -----------------------------------
# VERIFIED TARGET CCD codes
# -----------------------------------

TARGET_CCD_CODES: Dict[str, str] = {
        #Cocaine and cocaine metabolites
        "cocaine": "COC",
        "benzoylecgonine": "BCG", #Cocaine metabolite, found in urine

        #Opioids
        "morphine": "MOI",
        "fentanyl": "7V7",
        "oxycodone": "OOX",

        #Opioids antagonists
        "naloxone":"A1APV",
        "naltrexone":"A1CLJ",
        #Stimulants - Amphetamines
        "dextroamphetamine": "1WE", #THIS IS dextroamphetamine, aka (+)-amphetamine, there is also levoamphetamine aks (-)-amphetamine; the file only mentioend amphetamine
        "methamphetamine": "B40",
        "mdma": "B41",

        #Cannabinoids
        "thc": "TCI", #is also knoown as  Dronabinol and Marinol
        "cannabidiol": "P0T",



        #Benzodiazepines
        "alprazolam": "08H",
        "diazepam": "DZP",

        #Local anesthetics\
        "lidocaine": "LQZ",


        #Other adulterants and cutting agents
        "acetaminophen": "TYL",
        "caffeine": "CFF",
        "mannitol": "MTL",

        #Antidepressants / Antipsychotics,
        "fluoxetine": "SFX",
        "chlorpromazine": "Z80",
        "clomipramine": "CXX",
        "diphenhydramine": "2PM",

        #Alkaloids
        "quinine": "QI9",
        "papaverine": "EV1",
        "noscapine": "08N",
        "nicotine": "NCT",
        "scopolamine":"OW0",

        #NSAIDs
        "ibuprofen":"IBP",

        #Psychedelics / Tryptamines
        "lsd": "7LD", #Lysergic acid diethylamide

        #Neurotransmitters
        #domapine: LDP
        "serotonin": "SRO",


        #Amino acidssss
        "glycine": "GLY",
        "tyrosine": "TYR",
        "phenylalanine": "PHE",
        "tryptophan": "TRP",
        "tyramine": "AEF",

        #Steriods
        "dhea_sulfate": "ZWY",
        "p_hydroxymethamphetamine":  "1WF"





}

TARGET_SMILES: Dict[str,str] = {
        "methadone": "CCC(=O)C(CC(C)N(C)C)(C1=CC=CC=C1)C2=CC=CC=C2",
        "codeine": "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O",

        "levoamphetamine": "C[C@H](CC1=CC=CC=C1)N",
        "pseudoephedrine": "C[C@@H]([C@H](C1=CC=CC=C1)O)NC",
        "methylphenidate": "COC(=O)C(C1CCCCN1)C2=CC=CC=C2",
        "benzocaine": "CCOC(=O)C1=CC=C(C=C1)N",
        "procaine": "CCN(CC)CCOC(=O)C1=CC=C(C=C1)N",
        "levamisole": "C1CSC2=N[C@H](CN21)C3=CC=CC=C3",
        "clonazepam": "C1C(=O)NC2=C(C=C(C=C2)[N+](=O)[O-])C(=N1)C3=CC=CC=C3Cl",
        "lorazepam": "C1=CC=C(C(=C1)C2=NC(C(=O)NC3=C2C=C(C=C3)Cl)O)Cl",
        "cannabinol": "CCCCCC1=CC(=C2C(=C1)OC(C3=C2C=C(C=C3)C)(C)C)O",
        "citalopram": "CN(C)CCCC1(C2=C(CO1)C=C(C=C2)C#N)C3=CC=C(C=C3)F",
        "bupropion": "CC(C(=O)C1=CC(=CC=C1)Cl)NC(C)(C)C",
        "amoxapine": "C1CN(CCN1)C2=NC3=CC=CC=C3OC4=C2C=C(C=C4)Cl",
        "dmt": "CN(C)CCC1=CNC2=CC=CC=C21",
        "sumatriptan":"CNS(=O)(=O)CC1=CC2=C(C=C1)NC=C2CCN(C)C",
        "dopamine":"C1=CC(=C(C=C1CCN)O)O",
        "epinephrine":"CNC[C@@H](C1=CC(=C(C=C1)O)O)O",
        "norepinephrine":"C1=CC(=C(C=C1[C@H](CN)O)O)O",
        "gaba":"C(CC(=O)O)CN",#Gamma-aminobutyric acid (GABA) ,
        "homovanillic_acid":"COC1=C(C=CC(=C1)CC(=O)O)O",
        "dopac":"C1=CC(=C(C=C1CC(=O)O)O)O",#3,4-Dihydroxyphenylacetic acid
        "lactose":"C([C@@H]1[C@@H]([C@@H]([C@H]([C@@H](O1)O[C@H]([C@@H](CO)O)[C@@H]([C@H](C=O)O)O)O)O)O)O",
        "xylazine":"CC1=C(C(=CC=C1)C)NC2=NCCCS2",
        "heroin":"CC(=O)O[C@H]1C=C[C@H]2[C@H]3CC4=C5[C@]2([C@H]1OC5=C(C=C4)OC(=O)C)CCN3C",
        "normorphine":"C1CN[C@@H]2CC3=C4[C@@]15[C@H]2C=C[C@@H]([C@@H]5OC4=C(C=C3)O)O",
        "methylnaltrexone":"C[N+]1(CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)O)O4)O)CC6CC6",
        "tramadol":"CN(C)C[C@H]1CCCC[C@@]1(C2=CC(=CC=C2)OC)O",
        "n_desmethyl_cis_tramadol":"CNC[C@H]1CCCC[C@]1(C2=CC(=CC=C2)OC)O",
        "morphine_3_glucuronide":"CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O[C@H]6[C@@H]([C@H]([C@@H]([C@H](O6)C(=O)O)O)O)O)O[C@H]3[C@H](C=C4)O",
        "morphine_6_glucuronide":"CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O[C@H]6[C@@H]([C@H]([C@@H]([C@H](O6)C(=O)O)O)O)O",
        "acetyl_fentanyl":"CC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3",
        "furanyl_fentanyl":"C1CN(CCC1N(C2=CC=CC=C2)C(=O)C3=CC=CO3)CCC4=CC=CC=C4",
        "mephedrone":"CC1=CC=C(C=C1)C(=O)C(C)NC",
        "ethylone":"CCNC(C)C(=O)C1=CC2=C(C=C1)OCO2",
        "mdpv":"CCCC(C(=O)C1=CC2=C(C=C1)OCO2)N3CCCC3",
        "hydroxyamfetamine":"CC(CC1=CC=C(C=C1)O)N",
        "ab_fubinaca":"CC(C)[C@@H](C(=O)N)NC(=O)C1=NN(C2=CC=CC=C21)CC3=CC=C(C=C3)F",
        "5f_amb":"COC(C(C(C)C)NC(C1=NN(C2=C1C=CC=C2)CCCCCF)=O)=O",
        "ur_144":"CCCCCN1C=C(C2=CC=CC=C21)C(=O)C3C(C3(C)C)(C)C"


}



##
#Note to my self:
#So, I have two structs, an dictionaries to be exact. I need to itterate over it and add it. MAybe adding stuff it will be a later point
#Lets start with the CLI input commands, I want change pathing, generation of seeds, Ions, and all of that stuff.
#Maybe in the future, making the chemical compounds libs, would be nicer
##


#Filename, because usefull
file_path = __file__
file_name = os.path.basename(file_path)


#argparse CLI inputs andd such
parser = argparse.ArgumentParser(prog=f"{file_name}",description=f"""This script {file_name} generates local Alphafold3 with the possibilities for IONS being included""",usage='%(prog)s [options]', epilog="this ist still a WIP")
parser.add_argument('--aptamer_csv', required=True, help="this need to point towards the aptamer dataset")
parser.add_argument('--output_path', required=True, help="this need to point towards the output files")
parser.add_argument('--aptamer_class',required=True, help='type of aptamer')
parser.add_argument('--target', required=True, help='the target of the aptamer for the AF3 run')
parser.add_argument('--seed',help='the amount and which seeds')
parser.add_argument('--troubleshoot', action='store_true', help="toggling troubleshooting on")
parser.add_argument('--condition_label', required=True, help="label for file naming")
group = parser.add_mutually_exclusive_group()
group.add_argument('--ions', nargs='+', help="this takes the ion and the amount of ions, e.g. --ions Mg 2 Fe 5")


#Loadinput
def load_input(aptamer_csv_path, troubleshoot):
    df = pd.read_csv(aptamer_csv_path)
    if troubleshoot:
        print(f"df head: \n {df.head()}")
        print(f"df shape: {df.shape}")
        print(f"df columns: {df.columns}")
    return df


#select input
def select_aptamers(aptamer_df, aptamers_class, troubleshoot):
    if isinstance(aptamer_df, pd.DataFrame):
        cols = aptamer_df.columns
        if troubleshoot:
            print("\n ==================== DF Trouble =============== \n")
            print(f"shape: {aptamer_df.shape} \n")
            print(f"\n df head: \n {aptamer_df.head()}")
            print(f"aptamer_class: {aptamers_class}")
            print(f"aptamer columns: {cols}")
            print(f"target in cols: {aptamers_class} in {cols}")
            print("\n ==================== END DF Trouble ============\n")
        if 'target' in cols:
            df_subset = aptamer_df.loc[aptamer_df['target']== aptamers_class]

            #Sanity checks
            #SC1, check if multiple drugs gets selected or no drug
            if len(df_subset['target'].unique()) != 1:
                print(f"ERROR03-P1: No aptamers for this class drug OR multiple classes. Remember: drug targets are -> targets: {aptamer_df['target'].unique()}")
                print(f"ERROR03-P2: current subset is: \n {df_subset} \n")
                sys.exit(1)
            #SC2, check if there is duplicate sequences
            elif len(df_subset['sequence'].unique()) != df_subset.shape[0]:
                print(f"ERROR04: non unique sequences detected! Check and fix!")
                sys.exit(1)
            #SC3, check if there duplicate aptamer ids.
            elif len(df_subset['aptamer_id'].unique()) != df_subset.shape[0]:
                print(f"ERROR05: non unique aptamer ID detected! Check and fix")
                sys.exit(1)
        elif troubleshoot:
            print("ERROR02: target column missing in df")
            sys.exit(1)
        else:
            sys.exit(1)
    elif troubleshoot:
        print(f"ERROR01: Aptamer selection. Aptamer_df {aptamer_df}")
        sys.exit(1)
    else:
        sys.exit(1)
    return df_subset

#Create JSON, TODO: DNA_RNA needs to be added to input arguments
def build_af3_json(aptamer_id,DNA_RNA, sequence,target_label, target_entry, mg_count = 0,na_count = 0,cl_count = 0,K_count = 0,ca_count = 0,TRS_count = 0, condition_label = 0, seeds = None,troubleshoot = True, smiles = True):
    """
    building the AF3 json
    s, brick for brick


    #Important layout,
    #Index 0 = aptamaer sequence
    #Index 1 = target
    #Index 2+ = all other ions

    """
    if not aptamer_id or not sequence or not target_entry:
        print("\n =============== BIG ERROR JSON BUILDING ======================")
        print(f"aptamer_id: {aptamer_id}")
        print(f"sequence: {sequence}")
        print(f"target: {target_entry}")
        print("=================================================================")
        sys.exit(1)

    else:
        #Json that will get appended on the way
        json_df = []


        A = sequence.upper() #So, it technically should already be upper, but just to be sure
        A_label = DNA_RNA.upper()
        #Sanity check RNA has U in it and no T
        if "U" in A and DNA_RNA == "DNA":
            print(f"EORROR06-P1: U in DNA, seq: {A}, aptamer ID: {aptamer_id}")
            sys.exit(1)
    #TODO fix my dam shift ;: key, to precvent it moving in python, apparently is good in C

        elif "T" in A and DNA_RNA == "RNA":
            print(f"EORROR06-P2: T in RNA, seq: {A}, aptamer ID: {aptamer_id}")
            sys.exit(1)
        A_label = A_label.lower() #AF3 wants DNA to be dna, so lower case
        aptamer_entry = {A_label:{"id":"A","sequence":A}} #sequences need to be string not in a list
        if troubleshoot:
            print(f"aptamer_entry: {aptamer_entry}")

        json_df.append(aptamer_entry)



        #Writing B part
        if troubleshoot:
            print(f"target: {target_entry}")
        #sanity check
        if list(target_entry.values())[0].get("id") != "B":
            print("ERROR07: target_entry does not have chain ID B")
            print(f"target_entry: {target_entry}")
            sys.exit(1)
        json_df.append(target_entry)






        #Ions
        ions = []
        #selecting the letter C
        LP_boundry = 67
        UP_boundry = 67 #this will increase, in body

        if mg_count is not None and mg_count > 0:
            #Increasing counter
            #LB_boundry, will not be increased yet because Mg is first one
            UP_boundry = UP_boundry + mg_count
            mg_chains = [] #Dict stuct
            for i in range(LP_boundry, UP_boundry):
                mg_chains.append({"ligand":{"id":chr(i),"ccdCodes":["MG"]}})
            ions.extend(mg_chains)


        if na_count is not None and na_count > 0:
            #boundries
            LP_boundry = UP_boundry
            UP_boundry = UP_boundry + na_count
            na_chains = []
            for i in range(LP_boundry,UP_boundry):
                na_chains.append({"ligand":{"id":chr(i),"ccdCodes":["NA"]}})
            ions.extend(na_chains)

        if cl_count is not None and cl_count > 0:
           #boundries
            LP_boundry = UP_boundry
            UP_boundry = UP_boundry + cl_count
            cl_chains = []
            for i in range(LP_boundry,UP_boundry):
                cl_chains.append({"ligand":{"id":chr(i),"ccdCodes":["CL"]}})
            ions.extend(cl_chains)

        if K_count is not None and K_count > 0:
            #boundries
            LP_boundry = UP_boundry
            UP_boundry = UP_boundry + K_count
            K_chains = []
            for i in range(LP_boundry,UP_boundry):
                K_chains.append({"ligand":{"id":chr(i),"ccdCodes":["K"]}})
            ions.extend(K_chains)


        if ca_count is not None and ca_count > 0:
            #boundries
            LP_boundry = UP_boundry
            UP_boundry = UP_boundry + ca_count
            ca_chains = []
            for i in range(LP_boundry,UP_boundry):
                ca_chains.append({"ligand":{"id":chr(i),"ccdCodes":["CA"]}})
            ions.extend(ca_chains)

        if TRS_count is not None and TRS_count > 0:
            #boundries
            LP_boundry = UP_boundry
            UP_boundry = UP_boundry + TRS_count
            TRS_chains = []
            for i in range(LP_boundry,UP_boundry):
                TRS_chains.append({"ligand":{"id":chr(i),"ccdCodes":["TRS"]}})
            ions.extend(TRS_chains)
        #Ions added
        json_df.extend(ions)

        #File names saving, including ions part in file names
        ion_parts = [(label, count) for label, count in [
            ("MG", mg_count), ("NA", na_count), ("CL", cl_count),
            ("K", K_count), ("CA", ca_count), ("TRS", TRS_count)
            ] if count is not None and count > 0]

        ion_string = "_".join(f"{label}_{count}" for label, count in ion_parts)

        if smiles:
            #smiles have special names, will bork filename, propbaly should use trevialname in file_name, so need to add that to the TODO
            #TODO: use smiles names correctly for filename
            file_name = f"{aptamer_id}_Target_{target_label}_{condition_label}"
        else:
            #the target_entry is a dict, to amek future stuff easier, SO, need to grab the name
            ccdCodes_target = list(target_entry.values())[0].get("ccdCodes")[0]
            file_name = f"{aptamer_id}_Target_{ccdCodes_target}_{condition_label}"
        #moved ion_string here, otherwise when empty file it would
        if ion_string:
            file_name = f"{file_name}_{ion_string}"
        if seeds is None:
            seeds = [1,2,3,4,5]



        #Json creation
        json_data = {
                "name": file_name,
                "modelSeeds": seeds,
                "sequences": json_df,
                "bondedAtomPairs": None,
                "userCCD": None,
                "dialect": "alphafold3",
                "version": 4
        }

    return json_data




#TODO custom get_target_entity
def get_target_entity(target_CLI,target_ccds_lib, target_smiles_lib, troubleshoot, type_target = 'ligand'):
    target_CLI = target_CLI.lower()
    target_CLI = target_CLI.strip()

    if troubleshoot:
        print(f"inputted target CLI: {target_CLI}")
    if target_CLI in target_ccds_lib:
        target_entry = {type_target:{"id":"B","ccdCodes":[target_ccds_lib[target_CLI]]}}
        return target_entry, False #the fasle if for the smiles variable in main function
    elif target_CLI in target_smiles_lib:
        target_entry = {type_target:{"id":"B","smiles":target_smiles_lib[target_CLI]}}
        return target_entry, True #the True is for smiles variable in main function
    elif type_target != 'ligand':
        print("ERROR08: non-ligand targets is not yet supported")
        sys.exit(1)
    else:
        valid_targets = list(target_ccds_lib.keys()) + list(target_smiles_lib.keys())
        suggestions = difflib.get_close_matches(target_CLI,valid_targets, n=3, cutoff=0.6)
        print(f"ERROR09: not regonized target.you typed: {target_CLI} Did you mean: {suggestions}")
        sys.exit(1)






def parse_seed(seed_str:str) -> list[int]:
    clean_seed = []
    seed_string = seed_str.split(',')
    for element in seed_string:
        element = element.strip()
        if '-' in element:
            parts = element.split('-')
            start, end = int(parts[0]), int(parts[1])
            numbers = range(start,end +1) #+1 to makethe max range including
            clean_seed.extend(numbers)
        else:
            clean_seed.append(int(element))

    return sorted(set(clean_seed))





#main function
def main():
    args = parser.parse_args()

    #access CLI varaibles
    aptamer_csv_path = args.aptamer_csv
    output_path = args.output_path
    aptamer_class = args.aptamer_class
    target_CLI = args.target
    ions = args.ions
    condition_label = args.condition_label
    seeds = args.seed
    troubleshoot = args.troubleshoot

    #droping whitespace
    aptamer_csv_path = aptamer_csv_path.strip()
    output_path = output_path.strip()
    #DNA_RNA, should maybe make it more automatic
    dna_rna = "DNA"

    #sanity check
    if not os.path.isfile(aptamer_csv_path):
        print(f"ERROR10: path doesn't lead to an file. inputed path = {aptamer_csv_path}")
        sys.exit(1)
    aptamer_df = pd.read_csv(aptamer_csv_path)
    #geting subset
    #TODO: REMEMBER ADD smiles!!!:
    df_aptamers_subset = select_aptamers(aptamer_df, aptamer_class, troubleshoot)


    target_entry, smiles = get_target_entity(target_CLI, TARGET_CCD_CODES, TARGET_SMILES, troubleshoot) #TODO:  type_target is not yet included in the CLI

    #Seeedddsss
    seeds = parse_seed(seeds)


    #processing ions
    ion_map = {
    "Mg": "mg_count",
    "Na": "na_count",
    "Cl": "cl_count",
    "K":  "K_count",
    "Ca": "ca_count",
    "TRS": "TRS_count"
    }
    ion_counts = {"mg_count": 0, "na_count": 0, "cl_count": 0, "K_count": 0, "ca_count": 0, "TRS_count": 0}
    #prossesing seeds
    if ions:
        for i in range(0, len(ions),2):
            ion = ions[i].strip()
            count = int(ions[i+1])
            if ion not in ion_map:

                print(f"ERROR: unrecognized ion '{ion}'. Valid ions: {list(ion_map.keys())}")
                sys.exit(1)
            ion_counts[ion_map[ion]] = count

    for _, row in df_aptamers_subset.iterrows():
        #Should maybe but targets in her , so multulple itterations allowing
        json_data = build_af3_json(
            aptamer_id=row['aptamer_id'],
            sequence=row['sequence'],
            DNA_RNA=dna_rna,
            target_label=target_CLI,
            target_entry=target_entry,
            condition_label=condition_label,
            seeds=seeds,
            troubleshoot=troubleshoot,
            smiles=smiles,
            **ion_counts
        )
        #TODOL addoing saving stuff
        os.makedirs(os.path.join(output_path,condition_label), exist_ok=True)
        #TODO adding filename
        filename = f"{json_data['name']}.json"
        file_path = os.path.join(output_path,condition_label,filename)
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)





if __name__ == "__main__":
    main()
