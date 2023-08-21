import glob, ntpath, json, zipfile, os

MOLFILES = {}

# Specify the location of the 'start_conformers' directory
for sdf in glob.glob(os.path.join('..', 'start_conformers', '*.sdf')):
    MOLFILES[ntpath.basename(sdf).replace('.sdf', '')] = sdf

BANMOL = []

ARCHIVE_NAME = 'check_testset.zip'
JSON_NAME = "testcases.json"

if __name__ == "__main__":
    testcases = []
    for key in MOLFILES.keys():
        if key not in BANMOL:
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
