import os
import ROOT

from columns import tau_cond, reco_columns

module_path = os.path.join(os.path.dirname(__file__), "taus.h")

ROOT.gInterpreter.ProcessLine(f'#include "{module_path}"')


def jet_cleaning(d):
    cleaned = (
        d.Define("TMPGenElectronMask", "abs(GenPart_pdgId) == 11")
        .Define("TMPGenElectron_pt", "GenPart_pt[TMPGenElectronMask]")
        .Define("TMPGenElectron_eta", "GenPart_eta[TMPGenElectronMask]")
        .Define("TMPGenElectron_phi", "GenPart_phi[TMPGenElectronMask]")
        .Define("TMPGenMuonMask", "abs(GenPart_pdgId) == 13")
        .Define("TMPGenMuon_pt", "GenPart_pt[TMPGenMuonMask]")
        .Define("TMPGenMuon_eta", "GenPart_eta[TMPGenMuonMask]")
        .Define("TMPGenMuon_phi", "GenPart_phi[TMPGenMuonMask]")
        .Define("TMPGenTauMask", "abs(GenPart_pdgId) == 15")
        .Define("TMPGenTau_pt", "GenPart_pt[TMPGenTauMask]")
        .Define("TMPGenTau_eta", "GenPart_eta[TMPGenTauMask]")
        .Define("TMPGenTau_phi", "GenPart_phi[TMPGenTauMask]")
        .Define(
            "CleanGenJet_mask_ele",
            "clean_genjet_mask(GenJet_pt, GenJet_eta, GenJet_phi, TMPGenElectron_pt, TMPGenElectron_eta, TMPGenElectron_phi)",
        )
        .Define(
            "CleanGenJet_mask_muon",
            "clean_genjet_mask(GenJet_pt, GenJet_eta, GenJet_phi, TMPGenMuon_pt, TMPGenMuon_eta, TMPGenMuon_phi)",
        )
        .Define(
            "CleanGenJet_mask_tau",
            "clean_genjet_mask(GenJet_pt, GenJet_eta, GenJet_phi, TMPGenTau_pt, TMPGenTau_eta, TMPGenTau_phi)",
        )
        .Define("CleanGenJetMask", "CleanGenJet_mask_ele && CleanGenJet_mask_muon")
        .Define("CleanGenJet_pt", "GenJet_pt[CleanGenJetMask]")
        .Define("CleanGenJet_eta", "GenJet_eta[CleanGenJetMask]")
        .Define("CleanGenJet_phi", "GenJet_phi[CleanGenJetMask]")
        .Define("CleanGenJet_mass", "GenJet_mass[CleanGenJetMask]")
        .Define(
            "CleanGenJet_hadronFlavour_uchar", "GenJet_hadronFlavour[CleanGenJetMask]"
        )
        .Define(
            "CleanGenJet_hadronFlavour",
            "static_cast<ROOT::VecOps::RVec<int>>(CleanGenJet_hadronFlavour_uchar)",
        )
        .Define("CleanGenJet_partonFlavour", "GenJet_partonFlavour[CleanGenJetMask]")
    )

    return cleaned


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
            "match_reco_to_gen(Tau_genPartIdx, GenPart_statusFlags)",
        )
        .Define("TauMask", "TauIdxToLastCopy >=0")
        .Define("MatchedGenTaus", "Tau_genPartIdx[TauMask]")
        .Define("MGenTau_eta", "Take(GenPart_eta,MatchedGenTaus)")
        .Define("MGenTau_phi", "Take(GenPart_phi,MatchedGenTaus)")
        .Define("MGenTau_pt", "Take(GenPart_pt,MatchedGenTaus)")
        .Define("MGenTau_pdgId", "Take(GenPart_pdgId, MatchedGenTaus)")
        .Define("MGenTau_charge", "Mcharge(MGenTau_pdgId)")
        .Define("MGenPart_statusFlags", "Take(GenPart_statusFlags,MatchedGenTaus)")
        .Define("MGenPart_statusFlags0", "MBitwiseDecoder(MGenPart_statusFlags, 0)")
        .Define("MGenPart_statusFlags1", "MBitwiseDecoder(MGenPart_statusFlags, 1)")
        .Define("MGenPart_statusFlags2", "MBitwiseDecoder(MGenPart_statusFlags, 2)")
        .Define("MGenPart_statusFlags3", "MBitwiseDecoder(MGenPart_statusFlags, 3)")
        .Define("MGenPart_statusFlags4", "MBitwiseDecoder(MGenPart_statusFlags, 4)")
        .Define("MGenPart_statusFlags5", "MBitwiseDecoder(MGenPart_statusFlags, 5)")
        .Define("MGenPart_statusFlags6", "MBitwiseDecoder(MGenPart_statusFlags, 6)")
        .Define("MGenPart_statusFlags7", "MBitwiseDecoder(MGenPart_statusFlags, 7)")
        .Define("MGenPart_statusFlags8", "MBitwiseDecoder(MGenPart_statusFlags, 8)")
        .Define("MGenPart_statusFlags9", "MBitwiseDecoder(MGenPart_statusFlags, 9)")
        .Define("MGenPart_statusFlags10", "MBitwiseDecoder(MGenPart_statusFlags, 10)")
        .Define("MGenPart_statusFlags11", "MBitwiseDecoder(MGenPart_statusFlags, 11)")
        .Define("MGenPart_statusFlags12", "MBitwiseDecoder(MGenPart_statusFlags, 12)")
        .Define("MGenPart_statusFlags13", "MBitwiseDecoder(MGenPart_statusFlags, 13)")
        .Define("MGenPart_statusFlags14", "MBitwiseDecoder(MGenPart_statusFlags, 14)")
        .Define(
            "ClosestJet_dr",
            "Mclosest_jet_dr(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi)",
        )
        .Define(
            "ClosestJet_deta",
            "Mclosest_jet_deta(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi)",
        )
        .Define(
            "ClosestJet_dphi",
            "Mclosest_jet_dphi(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi)",
        )
        .Define(
            "ClosestJet_pt",
            "Mclosest_jet_pt(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_pt)",
        )
        .Define(
            "ClosestJet_mass",
            "Mclosest_jet_mass(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_mass)",
        )
        .Define(
            "ClosestJet_EncodedPartonFlavour_light",
            "closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{1,2,3})",
        )
        .Define(
            "ClosestJet_EncodedPartonFlavour_gluon",
            "closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{21})",
        )
        .Define(
            "ClosestJet_EncodedPartonFlavour_c",
            "closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{4})",
        )
        .Define(
            "ClosestJet_EncodedPartonFlavour_b",
            "closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{5})",
        )
        .Define(
            "ClosestJet_EncodedPartonFlavour_undefined",
            "closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_partonFlavour, ROOT::VecOps::RVec<int>{0})",
        )
        .Define(
            "ClosestJet_EncodedHadronFlavour_b",
            "closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_hadronFlavour, ROOT::VecOps::RVec<int>{5})",
        )
        .Define(
            "ClosestJet_EncodedHadronFlavour_c",
            "closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_hadronFlavour, ROOT::VecOps::RVec<int>{4})",
        )
        .Define(
            "ClosestJet_EncodedHadronFlavour_light",
            "closest_jet_flavour_encoder(CleanGenJet_eta, CleanGenJet_phi, MGenTau_eta, MGenTau_phi, CleanGenJet_hadronFlavour, ROOT::VecOps::RVec<int>{0})",
        )
        .Define("MTau_charge", "Tau_charge[TauMask]")
        .Define("MTau_cleanmask", "Tau_cleanmask[TauMask]")
        .Define("MTau_dxy", "Tau_dxy[TauMask]")
        .Define("MTau_dz", "Tau_dz[TauMask]")
        .Define("MTau_etaMinusGen", "Tau_eta[TauMask]-MGenTau_eta")
        .Define("MTau_filteredphi", "Tau_phi[TauMask]")
        .Define("MTau_phiMinusGen", "DeltaPhi(MTau_filteredphi, MGenTau_phi)")
        .Define("MTau_ptRatio", "Tau_pt[TauMask]/MGenTau_pt")
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
        .Define("MTau_idDeepTau2017v2p1VSmu", "Tau_idDeepTau2017v2p1VSmu[TauMask]")
        .Define("MTau_idAntiMu", "Tau_idAntiMu[TauMask]")
        .Define("MTau_idDecayModeOldDMs", "Tau_idDecayModeOldDMs[TauMask]")
        .Define("MTau_idAntiEleDeadECal", "Tau_idAntiEleDeadECal[TauMask]")           
    )
    return extracted


def extract_taus(inputname, outputname, dict):
    ROOT.EnableImplicitMT()

    print(f"Processing {inputname}...")

    d = ROOT.RDataFrame("Events", inputname)

    d = jet_cleaning(d)
    d = extractAllTauFeatures(d)

    n_match, n_reco = dict["RECOTAU_GENTAU"]

    n_match += d.Histo1D("MTau_ptRatio").GetEntries()
    n_reco += d.Histo1D("Tau_pt").GetEntries()

    dict["RECOTAU_GENTAU"] = (n_match, n_reco)

    cols = tau_cond + reco_columns

    d.Snapshot("MTaus", outputname, cols)

    print(f"{outputname} written")