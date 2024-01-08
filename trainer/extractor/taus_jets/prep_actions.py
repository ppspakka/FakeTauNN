import numpy as np

"""
Dictionary of preprocessing operations for conditioning and target variables.
It is generated make_dataset function. Values of dictionary are list objects in which
sepcify preprocessing operation. Every operation has the following template

                       ["string", *pars]

where "string" tells which operation to perform and *pars its parameters. Such operations are

saturation: ["s", [inf, sup]]
gaussian smearing: ["g", sigma, [inf, sup]]
transformation: ["t", func, [a, b]]  # func(a * x + b)

In the case of multiple operations, order follows the operation list indexing.
"""

target_dictionary_taus = {
    "MTau_dz": [["s", [-np.inf, 20]], ["t", np.arctan, [50, 0]]],
    "MTau_dxy": [["t", np.arctan, [150, 0]]],
    #"MTau_rawDeepTau2017v2p1VSmu",
    #"MTau_puCorr",
    "MTau_leadTkDeltaPhi": [
        ["t", np.arctan, [80, 0]],
    ],
    "MTau_leadTkDeltaEta": [
        ["t", np.arctan, [100, 0]],
    ],
    #"MTau_rawIso",
    #"MTau_chargedIso",
    #"MTau_rawDeepTau2017v2p1VSe",
    "MTau_rawIsodR03": [
        ["s", [-np.inf, 100]],
        ["t", np.log, [1, 0.00001]],
        ["gm", -11.51, 1, [-np.inf, -7.5]]
    ],
    #"MTau_rawDeepTau2017v2p1VSjet",
    "MTau_leadTkPtOverTauPt": [
        ["manual_range", [0.1, 5]],
        ["t", np.arctan, [10, -10]],
    ],
    "MTau_idDeepTau2017v2p1VSe": [
        ["t", np.log2, [1, 1]],
        ["u", 0.5, None]
    ],
    "MTau_idDeepTau2017v2p1VSjet": [
        ["t", np.log2, [1, 1]],
        ["u", 0.5, None]
    ],
    #"MTau_decayMode",
    "MTau_charge": [["u", 1, None]],
    "MTau_idDeepTau2017v2p1VSmu": [
        ["t", np.log2, [1, 1]],
        ["u", 0.5, None]
    ],
    "MTau_idAntiMu": [['u', 0.5, None]],
    "MTau_idDecayModeOldDMs": [["u", 0.5, None]],
    "MTau_idAntiEleDeadECal": [["u", 0.5, None]],
    "MTau_eta": [
        #["t", np.arctan, [100, 0]], # Recheck
        ["t", np.arcsin, [1/(2.61), 0]],
    ],
    "MTau_phi": [
        #["t", np.arctan, [80, 0]],
        ["t", np.arcsin, [1/(np.pi+0.001), 0]],
    ], 
    "MTau_pt" : [
        #["manual_range", [0.1, 5]],
        ["t", np.arctan, [10, -10]], # Recheck
    ],
    "MTau_cleanmask": [["u", 0.5, None]],
    "MTau_genPartFlav": [["u", 0.5, None]],
}
