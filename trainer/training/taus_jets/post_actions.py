import numpy as np

"""
Dictionary of postprocessing operations for conditioning and target variables.
It is generated make_dataset function. Values of dictionary are list objects in which
sepcify preprocessing operation. Every operation has the following template

                       ["string", *pars]

where "string" tells which operation to perform and *pars its parameters. Such operations are

unsmearing: ["d", [inf, sup]]
transformation: ["i", func, [a, b]]  # (func(x) - b) / a

In the case of multiple operations, order follows the operation list indexing.
"""

target_dictionary = {
 'dz': [["i", np.tan, [50, 0]]], 
 'dxy': [["i", np.tan, [150, 0]]], 
 'leadTkDeltaPhi': [
        ["i", np.tan, [80, 0]],
    ], 
 'leadTkDeltaEta': [
        ["i", np.tan, [100, 0]],
    ], 
 'rawIsodR03': [
        ["d", [-np.inf, -7.5], np.log(0.00001)],
        ["i", np.exp, [1, 0.00001]],
    ], 
 'leadTkPtOverTauPt': [
        ["i", np.tan, [10, -10]],
    ], 
 'idDeepTau2017v2p1VSe': [
     ["d", None, None],
     ["i", np.power, [1, 0], 2]
     ],
 'idDeepTau2017v2p1VSjet': [
     ["d", None, None],
     ["i", np.power, [1, 0], 2]
     ],
 'decayMode': [["udm"]],
 'charge': [["c", 0.0, [-1, 1]]], 
 'idDeepTau2017v2p1VSmu': [
     ["d", None, None],
     ["i", np.power, [1, 0], 2]
     ],
 'idAntiMu': [["d", None, None]],
 'idDecayModeOldDMs': [["d", None, None]],
 'idAntiEleDeadECal': [["d", None, None]],
 "etaMinusGen": [
        ["i", np.tan, [100, 0]],
    ], 
 "phiMinusGen": [
        ["i", np.tan, [80, 0]],
        ["pmp"],
    ], 
 'ptRatio': [
        ["i", np.tan, [10, -10]], 
    ], 
 'cleanmask': [["c", 0.5, [0, 1]]] 
}


target_dictionary_taus = {}
for key, value in target_dictionary.items():
    target_dictionary_taus["Tau_" + key] = value
