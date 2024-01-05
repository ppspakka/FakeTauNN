import os
import ROOT

from columns import jet_cond, reco_columns

module_path = os.path.join(os.path.dirname(__file__), "taus.h")

ROOT.gInterpreter.ProcessLine(f'#include "{module_path}"')


def extractAllTauFeatures(df):
    """for getting gentau, recotau and cleaned genjet features

    Args:
        df (rdataframe): original rdataframe (should be cleaned by jet copies)

    Returns:
        rdataframe: rdataframe with new features
    """
    extracted = (
         df.Define(
            "TauIdxToLastCopy",
            "match_reco_to_gen(Tau_jetIdx)",
        )
        .Define("TauMask", "TauIdxToLastCopy >= 0")
        .Define("MatchedJets", "Tau_jetIdx[TauMask]")
        .Define("MJet_area", "Take(Jet_area, MatchedJets)")
        .Define("MJet_bRegCorr", "Take(Jet_bRegCorr, MatchedJets)")
        .Define("MJet_bRegRes", "Take(Jet_bRegRes, MatchedJets)")
        .Define("MJet_btagCSVV2", "Take(Jet_btagCSVV2, MatchedJets)")
        .Define("MJet_btagDeepB", "Take(Jet_btagDeepB, MatchedJets)")
        .Define("MJet_btagDeepCvB", "Take(Jet_btagDeepCvB, MatchedJets)")
        .Define("MJet_btagDeepCvL", "Take(Jet_btagDeepCvL, MatchedJets)")
        .Define("MJet_btagDeepFlavB", "Take(Jet_btagDeepFlavB, MatchedJets)")
        .Define("MJet_btagDeepFlavCvB", "Take(Jet_btagDeepFlavCvB, MatchedJets)")
        .Define("MJet_btagDeepFlavCvL", "Take(Jet_btagDeepFlavCvL, MatchedJets)")
        .Define("MJet_btagDeepFlavQG", "Take(Jet_btagDeepFlavQG, MatchedJets)")
        .Define("MJet_cRegCorr", "Take(Jet_cRegCorr, MatchedJets)")
        .Define("MJet_cRegRes", "Take(Jet_cRegRes, MatchedJets)")
        .Define("MJet_chEmEF", "Take(Jet_chEmEF, MatchedJets)")
        .Define("MJet_chFPV0EF", "Take(Jet_chFPV0EF, MatchedJets)")
        .Define("MJet_chHEF", "Take(Jet_chHEF, MatchedJets)")
        .Define("MJet_cleanmask", "Take(Jet_cleanmask, MatchedJets)") # UChar check
        .Define("MJet_eta", "Take(Jet_eta, MatchedJets)")
        .Define("MJet_hadronFlavour", "Take(Jet_hadronFlavour, MatchedJets)") # UChar check
        .Define("MJet_hfadjacentEtaStripsSize", "Take(Jet_hfadjacentEtaStripsSize, MatchedJets)")
        .Define("MJet_hfcentralEtaStripSize", "Take(Jet_hfcentralEtaStripSize, MatchedJets)")
        .Define("MJet_hfsigmaEtaEta", "Take(Jet_hfsigmaEtaEta, MatchedJets)")
        .Define("MJet_hfsigmaPhiPhi", "Take(Jet_hfsigmaPhiPhi, MatchedJets)")
        .Define("MJet_jetId", "Take(Jet_jetId, MatchedJets)")
        .Define("MJet_mass", "Take(Jet_mass, MatchedJets)")
        .Define("MJet_muEF", "Take(Jet_muEF, MatchedJets)")
        .Define("MJet_muonSubtrFactor", "Take(Jet_muonSubtrFactor, MatchedJets)")
        .Define("MJet_nConstituents", "Take(Jet_nConstituents, MatchedJets)")
        .Define("MJet_nElectrons", "Take(Jet_nElectrons, MatchedJets)")
        .Define("MJet_nMuons", "Take(Jet_nMuons, MatchedJets)")
        .Define("MJet_neEmEF", "Take(Jet_neEmEF, MatchedJets)")
        .Define("MJet_neHEF", "Take(Jet_neHEF, MatchedJets)")
        .Define("MJet_partonFlavour", "Take(Jet_partonFlavour, MatchedJets)")
        .Define("MJet_phi", "Take(Jet_phi, MatchedJets)")
        .Define("MJet_pt", "Take(Jet_pt, MatchedJets)")
        .Define("MJet_puId", "Take(Jet_puId, MatchedJets)")
        .Define("MJet_puIdDisc", "Take(Jet_puIdDisc, MatchedJets)")
        .Define("MJet_qgl", "Take(Jet_qgl, MatchedJets)")
        .Define("MJet_rawFactor", "Take(Jet_rawFactor, MatchedJets)")
        .Define("MTau_dz", "Tau_dz[TauMask]") # Recheck
        .Define("MTau_dxy", "Tau_dxy[TauMask]")
        .Define("MTau_rawDeepTau2017v2p1VSmu", "Tau_rawDeepTau2017v2p1VSmu[TauMask]")
        .Define("MTau_puCorr", "Tau_puCorr[TauMask]")
        .Define("MTau_leadTkDeltaPhi", "Tau_leadTkDeltaPhi[TauMask]")
        .Define("MTau_leadTkDeltaEta", "Tau_leadTkDeltaEta[TauMask]")
        .Define("MTau_rawIso", "Tau_rawIso[TauMask]")
        .Define("MTau_chargedIso", "Tau_chargedIso[TauMask]")
        .Define("MTau_rawDeepTau2017v2p1VSe", "Tau_rawDeepTau2017v2p1VSe[TauMask]")
        .Define("MTau_rawIsodR03", "Tau_rawIsodR03[TauMask]")
        .Define("MTau_rawDeepTau2017v2p1VSjet", "Tau_rawDeepTau2017v2p1VSjet[TauMask]")
        .Define("MTau_leadTkPtOverTauPt", "Tau_leadTkPtOverTauPt[TauMask]")
        .Define("MTau_idDeepTau2017v2p1VSe", "Tau_idDeepTau2017v2p1VSe[TauMask]")
        .Define("MTau_idDeepTau2017v2p1VSjet", "Tau_idDeepTau2017v2p1VSjet[TauMask]")
        .Define("MTau_decayMode", "Tau_decayMode[TauMask]")
        .Define("MTau_charge", "Tau_charge[TauMask]")
        .Define("MTau_idDeepTau2017v2p1VSmu", "Tau_idDeepTau2017v2p1VSmu[TauMask]")
        .Define("MTau_idAntiMu", "Tau_idAntiMu[TauMask]")
        .Define("MTau_idDecayModeOldDMs", "Tau_idDecayModeOldDMs[TauMask]")
        .Define("MTau_idAntiEleDeadECal", "Tau_idAntiEleDeadECal[TauMask]")
        .Define("MTau_eta", "Tau_eta[TauMask]")
        .Define("MTau_filteredphi", "Tau_phi[TauMask]") # Recheck
        .Define("MTau_phi", "Tau_phi[TauMask]")         # Recheck
        .Define("MTau_pt", "Tau_pt[TauMask]")
        .Define("MTau_cleanmask", "Tau_cleanmask[TauMask]")
        .Define("MTau_genPartFlav", "Tau_genPartFlav[TauMask]")
    )

    return extracted

def extract_taus(inputname, outputname, dict):
    ROOT.EnableImplicitMT()

    print(f"Processing {inputname}...")

    d = ROOT.RDataFrame("Events", inputname)

    d = extractAllTauFeatures(d)

    n_match, n_reco = dict["RECOTAU_RECOJET"]

    n_match += d.Histo1D("MTau_pt").GetEntries()
    n_reco += d.Histo1D("Tau_pt").GetEntries()

    dict["RECOTAU_RECOJET"] = (n_match, n_reco)

    cols = jet_cond + reco_columns

    d.Snapshot("MTaus", outputname, cols)

    print(f"{outputname} written")