import glob, ntpath, time, json
import ringo

if __name__ == "__main__":
    isomorphism_timings = []
    # Performs short parallel conformation search for 3 test molecules
    for sdf_name in reversed(glob.glob('./release_assemble/test_systems/*.sdf')):
        print(f"Processing {sdf_name}")
        # Initialize Molecule object
        mol = ringo.Molecule(sdf=sdf_name) # , require_best_sequence=True
        
        # Create pool for future conformers
        p = ringo.Confpool()

        # Perform Monte-Carlo with generation time limit of 10 seconds
        mcr_kwargs = {
            'rmsd_settings': 'default',
            'nthreads': 4,
            'timelimit': 2,
        }

        start_time = time.time()
        results = ringo.run_confsearch(mol, pool=p, **mcr_kwargs)
        print(f"Generated {len(p)} conformers of {ntpath.basename(sdf_name)}")
        end_time = time.time()
        elapsed_time = end_time - start_time

        timings_item = {
            'mol': ntpath.basename(sdf_name).replace('.sdf', ''),
            'time': elapsed_time,
            'niso': p.get_num_isomorphisms()
        }
        isomorphism_timings.append(timings_item)
        print(repr(timings_item))

        # Clear feed for the next molecule
        ringo.clear_status_feed()

    # Remove temporary file
    ringo.cleanup()

    with open('niso_timins.json', "w") as f:
        json.dump(isomorphism_timings, f)
