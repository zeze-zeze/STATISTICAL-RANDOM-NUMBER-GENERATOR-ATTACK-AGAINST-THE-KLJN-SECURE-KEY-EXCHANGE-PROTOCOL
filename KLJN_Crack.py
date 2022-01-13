import numpy as np  

R_L = 10 # 10k ohm
R_H = 100 # 100k ohm
f_B = 500 # 500 HZ
T_eff = pow(10, 18) # 10^18 K
k = 1.38 * pow(10, -23) # Boltzmann Constant
M = 0.1

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

    for i in range(TIMESTAMPS * 8):
        noises.pop(0)

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


flag = ''.join([bin(ord(i))[2:].rjust(8, '0') for i in 'FLAG{cr4ck_KLJN_scheme_wi7h_stati5tic41_a774ck}'])

# Noise generation
TIMESTAMPS = 10
num = pow(2, 10)
noises = np.array([np.mean([np.random.normal() for i in range(num)]) for j in range(TIMESTAMPS * len(flag) * 8)])
noises = np.fft.fft(noises)
noises = [noises[i // 2] if i % 2 == 0 else 0 for i in range(len(noises) * 2)]
noises = np.fft.ifft(noises).real
noises = noises.tolist()

U_ws = []
U_HHs = []
U_LHs = []
U_HLs = []
U_LLs = []

for f in flag:
    if f == '0':
        _, _, _, _, _, _, _, _, _, _, _, _, _, U_w, U_HH, U_LH, U_HL, U_LL, _, _, _, _, _ = setup('LH', False)
    else:
        _, _, _, _, _, _, _, _, _, _, _, _, _, U_w, U_HH, U_LH, U_HL, U_LL, _, _, _, _, _ = setup('HH', False)

    U_ws.append(U_w)
    U_HHs.append(U_HH)
    U_LHs.append(U_LH)
    U_HLs.append(U_HL)
    U_LLs.append(U_LL)

f = open('data.py', 'w')
f.write('U_ws = ' + str(U_ws) + '\n\n')
f.write('U_HHs = ' + str(U_HHs) + '\n\n')
f.write('U_LHs = ' + str(U_LHs) + '\n\n')
f.write('U_HLs = ' + str(U_HLs) + '\n\n')
f.write('U_LLs = ' + str(U_LLs) + '\n\n')
f.close()
