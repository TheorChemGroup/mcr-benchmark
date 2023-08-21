import glob, ntpath, json, zipfile, os

MOLFILES = {}
# SELECTED_MOLECULES = ["pdb_1FKJ", "pdb_2IYA", "pdb_3KEE", "pdb_3M6G", "pdb_1A7X", "pdb_7UPJ", "pdb_2C6H", "pdb_2QZK", "pdb_1NWX", "pdb_1QZ5"]
SELECTED_MOLECULES = ["pdb_2QZK", "pdb_2C6H", "pdb_1NWX", "pdb_3M6G", "pdb_2IYA", "csd_FINWEE10", "csd_RULSUN", "csd_YIVNOG", "csd_MIWTER", "csd_RECRAT"]
for sdf in glob.glob(os.path.join('..', 'start_conformers', '*.sdf')):
    molname = ntpath.basename(sdf).replace('.sdf', '')
    if molname in SELECTED_MOLECULES:
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
    
    # This is just for visual check of molecules
    sdf_names = [fname for fname in final_testset.values()]
    with zipfile.ZipFile(ARCHIVE_NAME, 'w') as new_zip:
        # Add each file to the archive
        for filename in sdf_names:
            new_zip.write(filename)
