// utilities for TAU
#ifndef UTILS_TAU_H
#define UTILS_TAU_H

auto MDeltaPhi(ROOT::VecOps::RVec<float>& Phi1,
    ROOT::VecOps::RVec<float>& Phi2) {
    auto size = Phi1.size();
    ROOT::VecOps::RVec<float> dphis;
    dphis.reserve(size);
    for (size_t i = 0; i < size; i++) {
        Double_t dphi = TVector2::Phi_mpi_pi(Phi1[i] - Phi2[i]);
        dphis.emplace_back(dphi);
    }
    return dphis;
}
auto Mclosest_jet_dr(ROOT::VecOps::RVec<float>& etaj,
    ROOT::VecOps::RVec<float>& phij,
    ROOT::VecOps::RVec<float>& etam,
    ROOT::VecOps::RVec<float>& phim) {

    auto size_outer = etam.size();
    auto size_inner = etaj.size();
    ROOT::VecOps::RVec<float> distances;
    distances.reserve(size_outer);
    for (size_t i = 0; i < size_outer; i++) {
        distances.emplace_back(10);
        float closest = 1;
        for (size_t j = 0; j < size_inner; j++) {
            Double_t deta = etam[i] - etaj[j];
            Double_t dphi = TVector2::Phi_mpi_pi(phim[i] - phij[j]);
            float dr = TMath::Sqrt(deta * deta + dphi * dphi);
            if (dr < closest) {
                closest = dr;
            }
        }
        if (closest < 1) {
            distances[i] = closest;
        }
    }
    return distances;
}
auto Mclosest_jet_mass(ROOT::VecOps::RVec<float>& etaj,
    ROOT::VecOps::RVec<float>& phij,
    ROOT::VecOps::RVec<float>& etam,
    ROOT::VecOps::RVec<float>& phim,
    ROOT::VecOps::RVec<float>& massj) {

    auto size_outer = etam.size();
    auto size_inner = etaj.size();
    ROOT::VecOps::RVec<float> masses;
    masses.reserve(size_outer);
    for (size_t i = 0; i < size_outer; i++) {
        masses.emplace_back(-1);
        float closest = 1;
        for (size_t j = 0; j < size_inner; j++) {
            Double_t deta = etam[i] - etaj[j];
            Double_t dphi = TVector2::Phi_mpi_pi(phim[i] - phij[j]);
            float dr = TMath::Sqrt(deta * deta + dphi * dphi);
            if (dr < closest) {
                closest = dr;
                masses[i] = massj[j];
            }
        }
    }
    return masses;
}
auto Mclosest_jet_pt(ROOT::VecOps::RVec<float>& etaj,
    ROOT::VecOps::RVec<float>& phij,
    ROOT::VecOps::RVec<float>& etam,
    ROOT::VecOps::RVec<float>& phim,
    ROOT::VecOps::RVec<float>& ptj) {

    auto size_outer = etam.size();
    auto size_inner = etaj.size();
    ROOT::VecOps::RVec<float> pts;
    pts.reserve(size_outer);
    for (size_t i = 0; i < size_outer; i++) {
        pts.emplace_back(-1);
        float closest = 1;
        for (size_t j = 0; j < size_inner; j++) {
            Double_t deta = etam[i] - etaj[j];
            Double_t dphi = TVector2::Phi_mpi_pi(phim[i] - phij[j]);
            float dr = TMath::Sqrt(deta * deta + dphi * dphi);
            if (dr < closest) {
                closest = dr;
                pts[i] = ptj[j];
            }
        }
    }
    return pts;
}
auto Mclosest_jet_deta(ROOT::VecOps::RVec<float>& etaj,
    ROOT::VecOps::RVec<float>& phij,
    ROOT::VecOps::RVec<float>& etam,
    ROOT::VecOps::RVec<float>& phim) {

    auto size_outer = etam.size();
    auto size_inner = etaj.size();
    ROOT::VecOps::RVec<float> detas;
    detas.reserve(size_outer);
    for (size_t i = 0; i < size_outer; i++) {
        detas.emplace_back(10);
        float closest = 1;
        for (size_t j = 0; j < size_inner; j++) {
            Double_t deta = etam[i] - etaj[j];
            Double_t dphi = TVector2::Phi_mpi_pi(phim[i] - phij[j]);
            float dr = TMath::Sqrt(deta * deta + dphi * dphi);
            if (dr < closest) {
                closest = dr;
                detas[i] = deta;
            }
        }
    }
    return detas;
}
auto Mclosest_jet_dphi(ROOT::VecOps::RVec<float>& etaj,
    ROOT::VecOps::RVec<float>& phij,
    ROOT::VecOps::RVec<float>& etam,
    ROOT::VecOps::RVec<float>& phim) {

    auto size_outer = etam.size();
    auto size_inner = etaj.size();
    ROOT::VecOps::RVec<float> dphis;
    dphis.reserve(size_outer);
    for (size_t i = 0; i < size_outer; i++) {
        dphis.emplace_back(10);
        float closest = 1;
        for (size_t j = 0; j < size_inner; j++) {
            Double_t deta = etam[i] - etaj[j];
            Double_t dphi = TVector2::Phi_mpi_pi(phim[i] - phij[j]);
            float dr = TMath::Sqrt(deta * deta + dphi * dphi);
            if (dr < closest) {
                closest = dr;
                dphis[i] = dphi;
            }
        }
    }
    return dphis;
}
auto MBitwiseDecoder(ROOT::VecOps::RVec<int>& ints, int bit) {
    auto size = ints.size();
    ROOT::VecOps::RVec<float> bits;
    bits.reserve(size);
    int num = pow(2, (bit));
    for (size_t i = 0; i < size; i++) {
        Double_t bAND = ints[i] & num;
        if (bAND == num) {
            bits.emplace_back(1);
        }
        else {
            bits.emplace_back(0);
        }
    }
    return bits;
}

auto taus_per_event(ROOT::VecOps::RVec<int>& MGM) {
    int size = MGM.size();
    return size;
}

auto Mcharge(ROOT::VecOps::RVec<int>& pdgId) {
    auto size = pdgId.size();
    ROOT::VecOps::RVec<float> charge;
    charge.reserve(size);
    for (size_t i = 0; i < size; i++) {
        if (pdgId[i] == -15)
            charge.emplace_back(+1);
        else
            charge.emplace_back(-1);
    }
    return charge;
}

auto filter_taus(
    ROOT::VecOps::RVec<int>& Tau_jetIdx) {

    auto size = Tau_jetIdx.size();
    ROOT::VecOps::RVec<int> indexes(size);
    for (size_t i = 0; i < size; i++) {
        indexes[i] = -1;
        auto curr_index = Tau_jetIdx[i];
        if (curr_index >= 0) {
            indexes[i] = curr_index;
        }
    }
    return indexes;
}

auto clean_genjet_mask(ROOT::VecOps::RVec<float>& jet_pt,
    ROOT::VecOps::RVec<float>& jet_eta,
    ROOT::VecOps::RVec<float>& jet_phi,
    ROOT::VecOps::RVec<float>& lep_pt,
    ROOT::VecOps::RVec<float>& lep_eta,
    ROOT::VecOps::RVec<float>& lep_phi) {
    /* Mask to remove GenElectrons  and GenMuons from the GenJet collection.*/
    auto lep_size = lep_pt.size();
    auto jet_size = jet_pt.size();

    ROOT::VecOps::RVec<int> clean_jet_mask;
    clean_jet_mask.reserve(jet_size);

    for (size_t i = 0; i < jet_size; i++) {
        clean_jet_mask.push_back(1);
        for (size_t j = 0; j < lep_size; j++) {
            auto dpt = jet_pt[i] - lep_pt[j];
            auto deta = jet_eta[i] - lep_eta[j];
            auto dphi = TVector2::Phi_mpi_pi(jet_phi[i] - lep_phi[j]);
            auto dr = TMath::Sqrt(deta * deta + dphi * dphi);

            if ((dr <= 0.01) && ((dpt / lep_pt[j]) <= 0.001)) {
                clean_jet_mask[i] = 0;
            }
        }
    }
    return clean_jet_mask;
}

auto closest_jet_flavour_encoder(ROOT::VecOps::RVec<float>& etaj,
    ROOT::VecOps::RVec<float>& phij,
    ROOT::VecOps::RVec<float>& etae,
    ROOT::VecOps::RVec<float>& phie,
    ROOT::VecOps::RVec<int>& fj,
    ROOT::VecOps::RVec<int> flavours) {

    /* General function to encode the hadron and parton flavour of the closest
       Jet object. To be used for flavour one-hot encoding for training.
    */

    auto size_outer = etae.size();
    auto size_inner = etaj.size();
    auto n_flavours = flavours.size();
    ROOT::VecOps::RVec<int> fenc;
    fenc.reserve(size_outer);
    for (size_t i = 0; i < size_outer; i++) {
        fenc.emplace_back(0);
        float closest = 1;
        for (size_t j = 0; j < size_inner; j++) {
            Double_t deta = etae[i] - etaj[j];
            Double_t dphi = TVector2::Phi_mpi_pi(phie[i] - phij[j]);
            float dr = TMath::Sqrt(deta * deta + dphi * dphi);
            if (dr < closest) {
                closest = dr;
                for (size_t k = 0; k < n_flavours; k++) {
                    if (abs(fj[j]) == flavours[k]) {
                        fenc[i] = 1;
                    }
                }
            }
        }
    }
    return fenc;
}

#endif