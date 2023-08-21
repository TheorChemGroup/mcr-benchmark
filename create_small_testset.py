import glob, ntpath, json, zipfile, os

MOLFILES = {}

# 10 random middle-sized macrocycles selected
RANDOM_MOLECULES = ["pdb_2QZK", "pdb_2C6H", "pdb_1NWX", "pdb_3M6G", "pdb_2IYA", "csd_FINWEE10", "csd_RULSUN", "csd_YIVNOG", "csd_MIWTER", "csd_RECRAT"]

# Specify the location of the 'start_conformers' directory
for sdf in glob.glob(os.path.join('..', 'start_conformers', '*.sdf')):
    molname = ntpath.basename(sdf).replace('.sdf', '')
    if molname in RANDOM_MOLECULES:
        MOLFILES[molname] = sdf

ARCHIVE_NAME = 'check_small_testset.zip'
JSON_NAME = "small_testcases.json"

if __name__ == "__main__":
    testcases = []
    for key in MOLFILES.keys():
        testcases.append(key)

    # Finalize and do some checks (Assert that chose from MacroModel testset only)
    final_testset = {key: MOLFILES[key] for key in testcases}

    # Save for future use in other scripts
    with open(JSON_NAME, "w") as f:
        json.dump(final_testset, f)  
    
    # Save test molecules as zip
    sdf_names = [fname for fname in final_testset.values()]
    with zipfile.ZipFile(ARCHIVE_NAME, 'w') as new_zip:
        # Add each file to the archive
        for filename in sdf_names:
            new_zip.write(filename)
