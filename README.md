# Intro
This project is an unofficial implementation of [STATISTICAL RANDOM NUMBER GENERATOR ATTACK AGAINST THE KIRCHHOFF-LAW-JOHNSON-NOISE (KLJN) SECURE KEY EXCHANGE PROTOCOL](https://arxiv.org/ftp/arxiv/papers/2110/2110.03088.pdf). Not only the four attacks against KLJN scheme were implemented, but also a CTF challenge according to the attack was created. Make fun of it and issue (or PR :D) if there is any mistake.

The four attacks include
* Bilateral attack demonstration utilizing cross-correlations between Alice’s/Bob’s and Eve’s wire voltages, currents and powers
* Bilateral attack demonstration utilizing cross-correlations among the voltage sources
* Unilateral attack demonstration utilizing cross-correlations between Alice’s/Bob’s and Eve’s wire voltages, currents and powers
* Unilateral attack demonstration utilizing cross-correlations among the voltage sources

# Usage
This Project includes three file as follow:

## [Implementation.py](Implementation.py)
`python3 Implementation.py` can output the result under four attacks, and the default ground truth = LH and M = 0.1 .

## [KLJN_Crack.py](KLJN_Crack.py)
`python3 KLJN_Crack.py` will generate a CTF challenge file data.py, with which challenger can solve the challenge by the first attack.

## [sol.py](sol.py)
After executing KLJN_Crack.py, `python3 sol.py` will solve the challenge and output the flag.
By calculating cross-correlation coefficient of Uw, UHH, UHL, ULH, ULL, challenger can utilize the information leak and get the flag.
