import itertools
import random

# def get_chunks(xs, n):
#     n = max(1, n)
#     return list(xs[i:i+n] for i in range(0, len(xs), n))

def float_to_int(x):
    if x != int(x):
        return int(x) + 1
    else:
        return int(x)

def as_chunk_dict(elements, chunks):
    return {
        chunk_name: list(data)
        for chunk_name, data in zip(chunks, itertools.batched(elements, float_to_int(len(elements) / len(chunks))))
    }

if __name__ == "__main__":
    ALL_MACROMODEL_TESTCASES = {'csdAAGGAG10', 'csdCIYKUR', 'csdDIGTOC', 'csdICYSPA', 'csdNITFEB', 'csdRECRAT', 'csdWUMKIY', 'pdb1BXO', 'pdb1KHR', 'pdb1QPL', 'pdb2ASM', 'pdb2PH8', 'pdb3EKS', 'csdABOJOR', 'csdCIYLAY', 'csdFAGFEZ', 'csdIDIMAJ', 'csdNUCZUH', 'csdREMCOC', 'csdXACJAM', 'pdb1BZL', 'pdb1MRL', 'pdb1QZ5', 'pdb2ASO', 'pdb2Q0R', 'pdb3I6O', 'csdABOMOT', 'csdCLPGDH', 'csdFEGJAD', 'csdJINWUZ', 'csdOHIDUE', 'csdRULSUN', 'csdXAGXIN', 'pdb1E9W', 'pdb1NMK', 'pdb1QZ6', 'pdb2ASP', 'pdb2QZK', 'pdb3K5C', 'csdACICOE', 'csdCOHVAW', 'csdFINWEE10', 'csdJOKSAD', 'csdOHIFAM', 'csdRUQVAB', 'csdXAGXOT', 'pdb1EHL', 'pdb1NSG', 'pdb1S22', 'pdb2C6H', 'pdb2VYP', 'pdb3KEE', 'csdBIDPIN10', 'csdCTILIH10', 'csdFIVSAE', 'csdKAQJAN', 'csdOHUXEU', 'csdTERDAW', 'csdXAGYAG', 'pdb1ESV', 'pdb1NT1', 'pdb1SKX', 'pdb2C7X', 'pdb2VZM', 'pdb3M6G', 'csdBIHYAS10', 'csdCTVHVH01', 'csdFIXYEQ', 'csdKAQJER', 'csdPAPGAP', 'csdVAKPIH', 'csdYAGKOG', 'pdb1FKD', 'pdb1NTK', 'pdb1TPS', 'pdb2DG4', 'pdb2WEA', 'pdb7UPJ', 'csdCAHWEN', 'csdCUQRUB', 'csdGAFSEL', 'csdKIVDIC', 'csdPOXTRD10', 'csdVALINM30', 'csdYAGVEH', 'pdb1FKI', 'pdb1NWX', 'pdb1WUA', 'pdb2F3E', 'pdb2WHW', 'csdCALSAR', 'csdCYVIHV10', 'csdGEZBUI', 'csdLALWIF', 'csdQEKNID', 'csdVOKBIG', 'csdYIVNOG', 'pdb1FKJ', 'pdb1OSF', 'pdb1XBP', 'pdb2F3F', 'pdb2XBK', 'csdCAMVES01', 'csdDACMAW', 'csdGIHKOX10', 'csdMALVAL10', 'csdQOSNAN', 'csdVOLHOU', 'csdYUPBAN01', 'pdb1FKL', 'pdb1PFE', 'pdb1XS7', 'pdb2GPL', 'pdb3ABA', 'csdCAZVEF', 'csdDAFGIA', 'csdHAXMOI10', 'csdMECBOL', 'csdQUDRIQ', 'csdVUHHEM', 'csdYUYYEW', 'pdb1GHG', 'pdb1PKF', 'pdb1YET', 'pdb2IYA', 'pdb3BXR', 'csdCGPGAP10', 'csdDALPUC', 'csdHEBLIJ', 'csdMIWTER', 'csdQUDROW', 'csdWENFOL', 'csdZAJDUJ', 'pdb1JFF', 'pdb1PPJ', 'pdb1YND', 'pdb2IYF', 'pdb3DV1', 'csdCHPSAR', 'csdDEKSAN', 'csdHIKPAS01', 'csdMUCYIT', 'csdRASTIO', 'csdWUBGAB', 'pdb1A7X', 'pdb1KEG', 'pdb1QPF', 'pdb1YXQ', 'pdb2OTJ', 'pdb3DV5'}
    HEAVY_MACROMODEL_TESTCASES = {'csdFINWEE10', 'csdMIWTER', 'csdRECRAT', 'csdRULSUN', 'csdYIVNOG', 'pdb1NWX', 'pdb2C6H', 'pdb2IYA', 'pdb2QZK', 'pdb3M6G'}
    EASY_MACROMODEL_TESTCASES = ALL_MACROMODEL_TESTCASES - HEAVY_MACROMODEL_TESTCASES

    HEAVY_MACROMODEL_TESTCASES = list(HEAVY_MACROMODEL_TESTCASES)
    EASY_MACROMODEL_TESTCASES = list(EASY_MACROMODEL_TESTCASES)
    random.shuffle(HEAVY_MACROMODEL_TESTCASES)
    random.shuffle(EASY_MACROMODEL_TESTCASES)
    CHUNKS = ['A', 'C', 'D', 'X', 'Z']
    NUM_CHUNKS = len(CHUNKS)

    # easy_chunks = as_chunk_dict(EASY_MACROMODEL_TESTCASES, CHUNKS)
    # heavy_chunks = as_chunk_dict(HEAVY_MACROMODEL_TESTCASES, CHUNKS)
    
    # final_splitting = {
    #     chunk_name: easy_chunks[chunk_name] + heavy_chunks[chunk_name]
    #     for chunk_name in CHUNKS
    # }
    # final_splitting = {
    #     chunk_name: heavy_chunks[chunk_name]
    #     for chunk_name in CHUNKS
    #     if chunk_name in heavy_chunks
    # }

    chunks = as_chunk_dict(ALL_MACROMODEL_TESTCASES, CHUNKS)
    final_splitting = {
        chunk_name: chunks[chunk_name]
        for chunk_name in CHUNKS
    }

    check_list = [
        element
        for chunk_name, chunk_elements in final_splitting.items()
        for element in chunk_elements
    ]
    # assert set(check_list) == set(ALL_MACROMODEL_TESTCASES)
    # assert len(check_list) == len(ALL_MACROMODEL_TESTCASES)

    print(repr(final_splitting))
