import numpy as np  

# Noise generation
TIMESTAMPS = 10
num = pow(2, 10)
noises = np.array([np.mean([np.random.normal() for i in range(num)]) for j in range(TIMESTAMPS * 8)])
noises = np.fft.fft(noises)
noises = [noises[i // 2] if i % 2 == 0 else 0 for i in range(len(noises) * 2)]
noises = np.fft.ifft(noises).real
noises = noises.tolist()
assert len(noises) == 16 * TIMESTAMPS


R_L = 10 # 10k ohm
R_H = 100 # 100k ohm
f_B = 500 # 500 HZ
T_eff = pow(10, 18) # 10^18 K
k = 1.38 * pow(10, -23) # Boltzmann Constant
M = 1

def x(t):
    global noises
    return noises[t]

# Setup of Eve's noises
def setup(groundtruth, dummy=False):
    global R_L, R_H, f_B, T_eff, k, M, TIMESTAMPS

    U_EL_A = []
    U_star_EL_A = []
    U_L_A = []
    U_EH_A = []
    U_star_EH_A = []
    U_H_A = []
    U_EL_B = []
    U_star_EL_B = []
    U_L_B = []
    U_EH_B = []
    U_star_EH_B = []
    U_H_B = []

    AB_U_L_A = []
    AB_U_H_A = []
    AB_U_L_B = []
    AB_U_H_B = []

    for i in range(TIMESTAMPS):
        U_star_EL_A.append(x(0 * TIMESTAMPS + i) * np.sqrt(4 * k * T_eff * R_L * f_B))
        U_star_EH_A.append(x(1 * TIMESTAMPS + i) * np.sqrt(4 * k * T_eff * R_H * f_B))
        U_star_EL_B.append(x(2 * TIMESTAMPS + i) * np.sqrt(4 * k * T_eff * R_L * f_B))
        U_star_EH_B.append(x(3 * TIMESTAMPS + i) * np.sqrt(4 * k * T_eff * R_H * f_B))
        AB_U_L_A.append(x(4 * TIMESTAMPS + i) * np.sqrt(4 * k * T_eff * R_L * f_B))
        AB_U_H_A.append(x(5 * TIMESTAMPS + i) * np.sqrt(4 * k * T_eff * R_H * f_B))
        AB_U_L_B.append(x(6 * TIMESTAMPS + i) * np.sqrt(4 * k * T_eff * R_L * f_B))
        AB_U_H_B.append(x(7 * TIMESTAMPS + i) * np.sqrt(4 * k * T_eff * R_H * f_B))

    for i in range(TIMESTAMPS):
        U_EL_A.append(AB_U_L_A[i] + M * U_star_EL_A[i])
        U_EH_A.append(AB_U_H_A[i] + M * U_star_EH_A[i])
        U_EL_B.append(AB_U_L_B[i] + M * U_star_EL_B[i])
        U_EH_B.append(AB_U_H_B[i] + M * U_star_EH_B[i])

    for i in range(TIMESTAMPS):
        U_L_A.append(U_EL_A[i] / np.sqrt(np.sum(np.square(U_EL_A))) * np.sqrt(4 * k * T_eff * R_L * f_B))
        U_H_A.append(U_EH_A[i] / np.sqrt(np.sum(np.square(U_EH_A))) * np.sqrt(4 * k * T_eff * R_H * f_B))
        U_L_B.append(U_EL_B[i] / np.sqrt(np.sum(np.square(U_EL_B))) * np.sqrt(4 * k * T_eff * R_L * f_B))
        U_H_B.append(U_EH_B[i] / np.sqrt(np.sum(np.square(U_EH_B))) * np.sqrt(4 * k * T_eff * R_H * f_B))
    AB_U_L_A = np.array(AB_U_L_A) / np.sqrt(np.sum(np.square(AB_U_L_A))) * np.sqrt(4 * k * T_eff * R_L * f_B)
    AB_U_H_A = np.array(AB_U_H_A) / np.sqrt(np.sum(np.square(AB_U_H_A))) * np.sqrt(4 * k * T_eff * R_H * f_B)
    AB_U_L_B = np.array(AB_U_L_B) / np.sqrt(np.sum(np.square(AB_U_L_B))) * np.sqrt(4 * k * T_eff * R_L * f_B)
    AB_U_H_B = np.array(AB_U_H_B) / np.sqrt(np.sum(np.square(AB_U_H_B))) * np.sqrt(4 * k * T_eff * R_H * f_B)

    if dummy:
        AB_U_L_B = np.array([np.random.normal() for i in range(TIMESTAMPS)])
        AB_U_H_B = np.array([np.random.normal() for i in range(TIMESTAMPS)])

    I_w = [] # Eve measures
    I_HH = [] # Eve simulates
    I_LH = []
    I_HL = []
    I_LL = []
    U_w = [] # Eve measures
    U_HH = [] # Eve simulates
    U_LH = []
    U_HL = []
    U_LL = []
    P_w = [] # Eve measures
    P_HH = [] # Eve simulates
    P_LH = []
    P_HL = []
    P_LL = []

    for i in range(TIMESTAMPS):
        I_HH.append((AB_U_H_A[i] - AB_U_H_B[i]) / (R_H + R_H))
        I_LH.append((AB_U_L_A[i] - AB_U_H_B[i]) / (R_L + R_H))
        I_HL.append((AB_U_H_A[i] - AB_U_L_B[i]) / (R_H + R_L))
        I_LL.append((AB_U_L_A[i] - AB_U_L_B[i]) / (R_L + R_L))
        U_HH.append(I_HH[i] * R_H + AB_U_H_B[i])
        U_LH.append(I_LH[i] * R_H + AB_U_H_B[i])
        U_HL.append(I_HL[i] * R_L + AB_U_L_B[i])
        U_LL.append(I_LL[i] * R_L + AB_U_L_B[i])
        P_HH.append(U_HH[i] * I_HH[i])
        P_LH.append(U_LH[i] * I_LH[i])
        P_HL.append(U_HL[i] * I_HL[i])
        P_LL.append(U_LL[i] * I_LL[i])
        
        if groundtruth == 'HH':
            I_w.append((U_H_A[i] - U_H_B[i]) / (R_H + R_H))
            U_w.append(I_w[i] * R_H + U_H_B[i])
            P_w.append(U_w[i] * I_w[i])
        elif groundtruth == 'LH':
            I_w.append((U_L_A[i] - U_H_B[i]) / (R_L + R_H))
            U_w.append(I_w[i] * R_H + U_H_B[i])
            P_w.append(U_w[i] * I_w[i])
        elif groundtruth == 'HL':
            I_w.append((U_H_A[i] - U_L_B[i]) / (R_H + R_L))
            U_w.append(I_w[i] * R_L + U_L_B[i])
            P_w.append(U_w[i] * I_w[i])
        elif groundtruth == 'LL':
            I_w.append((U_L_A[i] - U_L_B[i]) / (R_L + R_L))
            U_w.append(I_w[i] * R_L + U_L_B[i])
            P_w.append(U_w[i] * I_w[i])
        else:
            print(f"Wrong groundtruth: {groundtruth}")
            raise

    return U_L_A, AB_U_L_A, U_H_A, AB_U_H_A, U_L_B, AB_U_L_B, U_H_B, AB_U_H_B, I_w, I_HH, I_LH, I_HL, I_LL, U_w, U_HH, U_LH, U_HL, U_LL, P_w, P_HH, P_LH, P_HL, P_LL

print(f'Ground Truth: LH, M: {M}')
U_L_A, AB_U_L_A, U_H_A, AB_U_H_A, U_L_B, AB_U_L_B, U_H_B, AB_U_H_B, I_w, I_HH, I_LH, I_HL, I_LL, U_w, U_HH, U_LH, U_HL, U_LL, P_w, P_HH, P_LH, P_HL, P_LL = setup('LH', False)

# Cross-correlation Coefficient
def CCC(comp1, comp2):
    assert len(comp1) == len(comp2)
    return (len(comp1) * np.sum(np.array(comp1) * np.array(comp2)) - np.sum(comp1) * np.sum(comp2)) / np.sqrt(((len(comp1) * np.sum(np.square(comp1)) - np.square(np.sum(comp1))) * (len(comp2) * np.sum(np.square(comp2)) - np.square(np.sum(comp2)))))

# Bilateral knowledge
## Cross-correlation attack utilizing Alice's/Bob's and Eve's channel voltages, currents and power
print("\nBilateral knowledge - Cross-correlation attack utilizing Alice's/Bob's and Eve's channel voltages, currents and power")

print(f"CCCu(HH, U): {CCC(U_HH, U_w)}, CCCi(HH, U): {CCC(I_HH, I_w)}, CCCp(HH, U): {CCC(P_HH, P_w)}")
print(f"CCCu(LH, U): {CCC(U_LH, U_w)}, CCCi(LH, U): {CCC(I_LH, I_w)}, CCCp(LH, U): {CCC(P_LH, P_w)}")
print(f"CCCu(HL, U): {CCC(U_HL, U_w)}, CCCi(HL, U): {CCC(I_HL, I_w)}, CCCp(HL, U): {CCC(P_HL, P_w)}")
print(f"CCCu(LL, U): {CCC(U_LL, U_w)}, CCCi(LL, U): {CCC(I_LL, I_w)}, CCCp(LL, U): {CCC(P_LL, P_w)}")

## Cross-correlation attack directly utilizing Alice's/Bob's and Eve's voltage sources
print("\nBilateral knowledge - Cross-correlation attack directly utilizing Alice's/Bob's and Eve's voltage sources")

print(f"CCCu(U*L_A, U_L_A): {CCC(U_L_A, AB_U_L_A)}")
print(f"CCCu(U*L_A, U_H_A): {CCC(U_L_A, AB_U_H_A)}")
print(f"CCCu(U*L_A, U_L_B): {CCC(U_L_A, AB_U_L_B)}")
print(f"CCCu(U*L_A, U_H_B): {CCC(U_L_A, AB_U_H_B)}")


# Unilateral knowledge
## Cross-correlation attack utilizing Alice's/Bob's and Eve's channel voltages, currents and power
print("\nUnilateral knowledge - Cross-correlation attack utilizing Alice's/Bob's and Eve's channel voltages, currents and power")

U_L_A, AB_U_L_A, U_H_A, AB_U_H_A, U_L_B, AB_U_L_B, U_H_B, AB_U_H_B, I_w, I_HH, I_LH, I_HL, I_LL, U_w, U_HH, U_LH, U_HL, U_LL, P_w, P_HH, P_LH, P_HL, P_LL = setup('LH', True)
print(f"CCCu(HH, U): {CCC(U_HH, U_w)}, CCCi(HH, U): {CCC(I_HH, I_w)}, CCCp(HH, U): {CCC(P_HH, P_w)}")
print(f"CCCu(LH, U): {CCC(U_LH, U_w)}, CCCi(LH, U): {CCC(I_LH, I_w)}, CCCp(LH, U): {CCC(P_LH, P_w)}")
print(f"CCCu(HL, U): {CCC(U_HL, U_w)}, CCCi(HL, U): {CCC(I_HL, I_w)}, CCCp(HL, U): {CCC(P_HL, P_w)}")
print(f"CCCu(LL, U): {CCC(U_LL, U_w)}, CCCi(LL, U): {CCC(I_LL, I_w)}, CCCp(LL, U): {CCC(P_LL, P_w)}")

## Cross-correlation attack utilizing Alice's and Eve's voltage sources
print("\nUnilateral knowledge - Cross-correlation attack utilizing Alice's and Eve's voltage sources")

print(f"CCCu(U*L_A, U_L_A): {CCC(U_L_A, AB_U_L_A)}")
print(f"CCCu(U*L_A, U_H_A): {CCC(U_L_A, AB_U_H_A)}")
