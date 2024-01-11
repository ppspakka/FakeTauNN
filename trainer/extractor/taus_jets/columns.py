# conditioning and reco columns for taus

jet_cond = [
    "MJet_area",
    "MJet_bRegCorr",
    "MJet_bRegRes",
    "MJet_btagCSVV2",
    "MJet_btagDeepB",
    "MJet_btagDeepCvB",
    "MJet_btagDeepCvL",
    "MJet_btagDeepFlavB",
    "MJet_btagDeepFlavCvB",
    "MJet_btagDeepFlavCvL",
    "MJet_btagDeepFlavQG",
    "MJet_cRegCorr",
    "MJet_cRegRes",
    "MJet_chEmEF",
    "MJet_chFPV0EF",
    "MJet_chHEF",
    "MJet_cleanmask",
    "MJet_eta",         #
    "MJet_hadronFlavour",
    "MJet_hfadjacentEtaStripsSize",
    "MJet_hfcentralEtaStripSize",
    "MJet_hfsigmaEtaEta",
    "MJet_hfsigmaPhiPhi",
    "MJet_jetId",
    "MJet_mass",
    "MJet_muEF",
    "MJet_muonSubtrFactor",
    "MJet_nConstituents",
    "MJet_nElectrons",
    "MJet_nMuons",
    "MJet_neEmEF",
    "MJet_neHEF",
    "MJet_partonFlavour",
    "MJet_phi",         #
    "MJet_pt",
    "MJet_puId",
    "MJet_puIdDisc",
    "MJet_qgl",
    "MJet_rawFactor",
]


tau_names = [
    "dz",
    "dxy",
    "rawDeepTau2017v2p1VSmu",
    "puCorr",
    "leadTkDeltaPhi",
    "leadTkDeltaEta",
    "rawIso",
    "chargedIso",
    "rawDeepTau2017v2p1VSe",
    "rawIsodR03",
    "rawDeepTau2017v2p1VSjet",
    "leadTkPtOverTauPt",
    "idDeepTau2017v2p1VSe",
    "idDeepTau2017v2p1VSjet",
    "decayMode",
    "charge",
    "idDeepTau2017v2p1VSmu",
    "idAntiMu",
    "idDecayModeOldDMs",
    "idAntiEleDeadECal",
    "etaMinusReco",          #
    "phiMinusReco",          #
    "ptRatio",          #   
    "cleanmask",
    "genPartFlav",
 ]

reco_columns = [f"MTau_{name}" for name in tau_names]