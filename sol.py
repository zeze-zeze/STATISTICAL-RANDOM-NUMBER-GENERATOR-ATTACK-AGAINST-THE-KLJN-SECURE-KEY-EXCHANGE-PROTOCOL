import numpy as np  
from data import *

assert len(U_ws) == len(U_HHs) == len(U_LHs) == len(U_HLs) == len(U_LLs)

# Cross-correlation Coefficient
def CCC(comp1, comp2):
    assert len(comp1) == len(comp2)
    return (len(comp1) * np.sum(np.array(comp1) * np.array(comp2)) - np.sum(comp1) * np.sum(comp2)) / np.sqrt(((len(comp1) * np.sum(np.square(comp1)) - np.square(np.sum(comp1))) * (len(comp2) * np.sum(np.square(comp2)) - np.square(np.sum(comp2)))))


encoded = ''
for i in range(len(U_ws)):
    if max(CCC(U_ws[i], U_HHs[i]), CCC(U_ws[i], U_HLs[i])) < max(CCC(U_ws[i], U_LHs[i]), CCC(U_ws[i], U_LLs[i])):
        encoded += '0'
    else:
        encoded += '1'

flag = ''
for i in range(0, len(encoded), 8):
    flag += chr(int(''.join([j for j in encoded[i:i+8]]), 2))
print(flag)

# flag: FLAG{cr4ck_KLJN_scheme_wi7h_stati5tic41_a774ck}
