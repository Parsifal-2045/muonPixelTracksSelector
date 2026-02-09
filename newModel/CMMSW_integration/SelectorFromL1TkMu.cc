#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <unordered_set>

/** class MuonTracksSelectorFromL1TkMuon
 *
 *   \author Luca Ferragina (INFN Bologna)
 */

class MuonTracksSelectorFromL1TkMuon : public edm::stream::EDProducer<> {
public:
  MuonTracksSelectorFromL1TkMuon(const edm::ParameterSet& iConfig);

  ~MuonTracksSelectorFromL1TkMuon() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // Tokens
  const edm::EDGetTokenT<l1t::TrackerMuonCollection> l1TkMuToken_;
  const edm::EDGetTokenT<reco::TrackCollection> tkToken_;
  // Track selection parameters
  const double l1TkMuMinPt_;
  const double tkMinPt_;
  const double tkMaxEta_;
  const double maxDz_;
  const double maxDr_;
  const double maxChi2_;
};

MuonTracksSelectorFromL1TkMuon::MuonTracksSelectorFromL1TkMuon(const edm::ParameterSet& iConfig)
    : l1TkMuToken_{consumes<l1t::TrackerMuonCollection>(
          iConfig.getParameter<edm::InputTag>("L1TkMuonInputCollection"))},
      tkToken_{consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("TrackInputCollection"))},
      l1TkMuMinPt_{iConfig.getParameter<double>("L1TkMuMinPt")},
      tkMinPt_{iConfig.getParameter<double>("trackMinPt")},
      tkMaxEta_{iConfig.getParameter<double>("trackMaxEta")},
      maxDz_{iConfig.getParameter<double>("maxDz")},
      maxDr_{iConfig.getParameter<double>("maxDr")},
      maxChi2_{iConfig.getParameter<double>("maxChi2")} {
  produces<reco::TrackCollection>();
}

#ifndef BASELINE_SELECTOR
void MuonTracksSelectorFromL1TkMuon::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string metname = "RecoMuon|L3TrackFinder|MuonTracksSelectorFromL1TkMuon";

  // Output collection
  std::unique_ptr<reco::TrackCollection> outputTracks = std::make_unique<reco::TrackCollection>();

  // Get input collections
  auto const l1TkMuCollectionH = iEvent.getHandle(l1TkMuToken_);
  auto const tkCollectionH = iEvent.getHandle(tkToken_);
  if (!l1TkMuCollectionH.isValid() || !tkCollectionH.isValid() || l1TkMuCollectionH->empty() ||
      tkCollectionH->empty()) {
    LogDebug(metname) << "Input collection not valid or empty! Returning empty collection." << '\n';
    iEvent.put(std::move(outputTracks));
    //std::cout << "MuonTracksSelectorFromL1TkMuon: Input collection not valid or empty! Returning empty collection."
    //          << '\n';
    return;
  }

  // Store indices of matched tracks
  std::set<size_t> tracksToKeep;
  size_t nTracksToKeep = 1;

  // Loop over L1TkMuons
  for (size_t l1TkMuIndex = 0; l1TkMuIndex != l1TkMuCollectionH->size(); ++l1TkMuIndex) {
    l1t::TrackerMuonRef l1TkMuRef(l1TkMuCollectionH, l1TkMuIndex);
    float l1TkMuEta, l1TkMuPhi, l1TkMuPt, l1TkMuZ0;
    auto trkPtr = l1TkMuRef->trkPtr();

    if (!trkPtr.isNull()) {
      // Prefer tracker-based coordinates
      l1TkMuEta = trkPtr->momentum().eta();
      l1TkMuPhi = trkPtr->momentum().phi();
      l1TkMuPt = trkPtr->momentum().perp();
      l1TkMuZ0 = trkPtr->z0();
    } else {
      // Fallback to standalone muon coordinates
      //std::cout << "L1TkMuon has no tracker pointer, using muon system coordinates";
      l1TkMuEta = l1TkMuRef->phEta();
      l1TkMuPhi = l1TkMuRef->phPhi();
      l1TkMuPt = l1TkMuRef->phPt();
      l1TkMuZ0 = l1TkMuRef->phZ0();
    }

    // Basic kinematic selection
    if (l1TkMuPt < l1TkMuMinPt_) {
      //std::cout << "MuonTracksSelectorFromL1TkMuon: L1Tk muon with pT = " << l1TkMuPt << " and eta = " << l1TkMuEta
      //          << " fails kinematic selection\n";
      continue;
    }

    // Pair of quality, track index for all candidate matches
    std::vector<std::pair<float, size_t>> potentialMatches;

    for (size_t tkIndex = 0; tkIndex != tkCollectionH->size(); ++tkIndex) {
      reco::TrackRef tkRef(tkCollectionH, tkIndex);

      // Loose pre-selection
      float dZ = std::abs(tkRef->dz() - l1TkMuZ0);
      if (dZ > 10.0)
        continue;

      float dR2 = deltaR2(l1TkMuEta, l1TkMuPhi, tkRef->eta(), tkRef->phi());
      if (dR2 > 0.5 * 0.5)
        continue;

      float l1TkMuQoverPt = float(l1TkMuRef->phCharge()) / l1TkMuPt;
      float tkQoverPt = float(tkRef->charge()) / tkRef->pt();
      float tkQoverPtError = tkRef->ptError() / (tkRef->pt() * tkRef->pt());
      if (tkQoverPtError < 1e-12)
        tkQoverPtError = 1e-12;  // Protection

      float curvDiff = l1TkMuQoverPt - tkQoverPt;
      float chi2Curv = (curvDiff * curvDiff) / (tkQoverPtError * tkQoverPtError);

      float termDr = dR2 / (maxDr_ * maxDr_);        // (dR / maxDr)^2
      float termDz = (dZ * dZ) / (maxDz_ * maxDz_);  // (dZ / maxDz)^2
      float termChi2 = chi2Curv / maxChi2_;          // chi2 / maxChi2

      float combinedQuality = termDr + termDz + termChi2;

      if (combinedQuality < 25) {
        potentialMatches.emplace_back(combinedQuality, tkIndex);
      }
    }  // End Track Loop

    // Sort by quality (lowest is best)
    std::partial_sort(potentialMatches.begin(),
                      potentialMatches.begin() + std::min(nTracksToKeep, potentialMatches.size()),
                      potentialMatches.end());

    for (size_t i = 0; i < std::min(potentialMatches.size(), nTracksToKeep); ++i) {
      reco::TrackRef tkRef(tkCollectionH, potentialMatches[i].second);

      tracksToKeep.insert(potentialMatches[i].second);
      /*
      std::cout << "MuonTracksSelectorFromL1TkMuon: L1Tk muon with index " << l1TkMuIndex
                << " matched to track with index " << potentialMatches[i].second
                << " with combinedQuality = " << potentialMatches[i].first << '\n';
      std::cout << "    -> dR = " << std::sqrt(deltaR2(l1TkMuEta, l1TkMuPhi, tkRef->eta(), tkRef->phi()))
                << ", chi2 qOverPt = "
                << (float(l1TkMuRef->phCharge()) / l1TkMuPt - float(tkRef->charge()) / tkRef->pt()) *
                       (float(l1TkMuRef->phCharge()) / l1TkMuPt - float(tkRef->charge()) / tkRef->pt()) /
                       (tkRef->ptError() / (tkRef->pt() * tkRef->pt()) * tkRef->ptError() / (tkRef->pt() * tkRef->pt()))
                << ", dZ =  " << std::abs(tkRef->dz() - l1TkMuZ0) << '\n';
      std::cout << "   L1TkMu: pT = " << l1TkMuPt << ", eta = " << l1TkMuEta << ", phi = " << l1TkMuPhi
                << ", z0 = " << l1TkMuZ0 << '\n';
      std::cout << "   Track:  pT = " << tkRef->pt() << ", eta = " << tkRef->eta() << ", phi = " << tkRef->phi()
                << ", z0 = " << tkRef->dz() << '\n';
      std::cout << "------------------------------------------------------" << '\n';
    */
    }
  }  // End L1TkMuon Loop

  outputTracks->reserve(tracksToKeep.size());
  for (size_t idx : tracksToKeep) {
    reco::TrackRef tkRef(tkCollectionH, idx);
    outputTracks->push_back(*tkRef);
  }

  iEvent.put(std::move(outputTracks));
}
#endif

#ifdef BASELINE_SELECTOR
void MuonTracksSelectorFromL1TkMuon::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string metname = "RecoMuon|L3TrackFinder|MuonTracksSelectorFromL1TkMuon";

  // Output collection
  std::unique_ptr<reco::TrackCollection> outputTracks = std::make_unique<reco::TrackCollection>();

  // Get input collections
  auto const l1TkMuCollectionH = iEvent.getHandle(l1TkMuToken_);
  auto const tkCollectionH = iEvent.getHandle(tkToken_);
  if (!l1TkMuCollectionH.isValid() || !tkCollectionH.isValid() || l1TkMuCollectionH->empty() ||
      tkCollectionH->empty()) {
    LogDebug(metname) << "Input collection not valid or empty! Returning empty collection." << '\n';
    iEvent.put(std::move(outputTracks));
    //std::cout << "MuonTracksSelectorFromL1TkMuon: Input collection not valid or empty! Returning empty collection."
    //          << '\n';
    return;
  }

  // Keep track of already selected tracks to avoid duplicates
  std::unordered_set<size_t> selectedTracksIndices;
  selectedTracksIndices.reserve(l1TkMuCollectionH->size() * 2);  // rough estimate

  outputTracks->reserve(l1TkMuCollectionH->size() * 2);  // rough estimate

  //std::cout << "MuonTracksSelectorFromL1TkMuon: Starting selection of tracks from " << tkCollectionH->size()
  //          << " input tracks and " << l1TkMuCollectionH->size() << " L1Tk muons" << '\n';
  // Loop over L1TkMuons
  for (size_t l1TkMuIndex = 0; l1TkMuIndex != l1TkMuCollectionH->size(); ++l1TkMuIndex) {
    int skippedZ = 0;
    int skippedDr = 0;
    int skippedPt = 0;
    /*
    bool skipL1TkMu = true;

    for (auto stub : l1TkMuCollectionH->at(l1TkMuIndex).stubs()) {
      if (!stub.isNull()) {
        if (stub->type() == 0) {
          if (stub->quality() >= 2) {
            skipL1TkMu = false;
            break;
          }
        } else if (stub->type() == 1) {
          if (stub->quality() >= 3) {
            skipL1TkMu = false;
            break;
          }
        } else {
          edm::LogError("MuonTracksSelectorFromL1TkMuon") << "Encountered stub with unknown type " << stub->type()
                                                               << " in L1TkMuon with index " << l1TkMuIndex << "\n";
        }
      }
    }

    if (skipL1TkMu) {
      std::cout << "MuonTracksSelectorFromL1TkMuon: Skipping L1Tk muon with index " << l1TkMuIndex
                << " since it has no stubs with high enough quality\n";
      continue;
    }
    */

    l1t::TrackerMuonRef l1TkMuRef(l1TkMuCollectionH, l1TkMuIndex);
    float l1TkMuEta, l1TkMuPhi, l1TkMuPt, l1TkMuZ0;
    auto trkPtr = l1TkMuRef->trkPtr();

    if (!trkPtr.isNull()) {
      // Prefer tracker-based coordinates
      l1TkMuEta = trkPtr->momentum().eta();
      l1TkMuPhi = trkPtr->momentum().phi();
      l1TkMuPt = trkPtr->momentum().perp();
      l1TkMuZ0 = trkPtr->z0();
    } else {
      // Fallback to standalone muon coordinates
      std::cout << "L1TkMuon has no tracker pointer, using muon system coordinates";
      l1TkMuEta = l1TkMuRef->phEta();
      l1TkMuPhi = l1TkMuRef->phPhi();
      l1TkMuPt = l1TkMuRef->phPt();
      l1TkMuZ0 = l1TkMuRef->phZ0();
    }

    // Basic kinematic selection
    if (l1TkMuPt < l1TkMuMinPt_) {
      std::cout << "MuonTracksSelectorFromL1TkMuon: L1Tk muon with pT = " << l1TkMuPt << " and eta = " << l1TkMuEta
                << " fails kinematic selection\n";
      continue;
    }
    // Loop over tracks
    for (size_t tkIndex = 0; tkIndex != tkCollectionH->size(); ++tkIndex) {
      // Skip already selected tracks
      if (selectedTracksIndices.contains(tkIndex)) {
        continue;  // Track already selected
      }

      reco::TrackRef tkRef(tkCollectionH, tkIndex);
      // Basic kinematic selection
      if (tkRef->pt() < tkMinPt_ || std::abs(tkRef->eta()) > tkMaxEta_) {
        continue;
      }

      // Check impact parameter match to reject PU tracks
      if (std::abs(tkRef->dz() - l1TkMuZ0) > maxDz_) {
        ++skippedZ;
        continue;
      }

      // Check match in dR
      float dR2 = deltaR2(l1TkMuEta, l1TkMuPhi, tkRef->eta(), tkRef->phi());
      if (dR2 < maxDr_ * maxDr_) {
#if 1
        int l1TkMuCharge = l1TkMuRef->phCharge();
        float l1TkMuQoverPt = float(l1TkMuCharge) / l1TkMuPt;

        float tkQoverPt = float(tkRef->charge()) / tkRef->pt();

        // sigma(1/x) ~ sigma(x) / x^2
        float tkQoverPtError = tkRef->ptError() / (tkRef->pt() * tkRef->pt());

        float curvDiff = l1TkMuQoverPt - tkQoverPt;
        float chi2Curv = (curvDiff * curvDiff) / (tkQoverPtError * tkQoverPtError);

        if (chi2Curv < maxChi2_) {
          selectedTracksIndices.insert(tkIndex);
          outputTracks->push_back(*tkRef);
          std::cout << "Track with index " << tkIndex << " matched to L1Tk muon ---  dR = " << sqrt(dR2)
                    << ", chi2 qOverPt = " << chi2Curv << ", dZ =  " << std::abs(tkRef->dz() - l1TkMuZ0) << '\n';
          std::cout << "   L1TkMu: pT = " << l1TkMuPt << ", eta = " << l1TkMuEta << ", phi = " << l1TkMuPhi
                    << ", z0 = " << l1TkMuZ0 << '\n';
          std::cout << "   Track:  pT = " << tkRef->pt() << ", eta = " << tkRef->eta() << ", phi = " << tkRef->phi()
                    << ", z0 = " << tkRef->dz() << '\n';
        } else {
          ++skippedPt;
        }
#endif
#if 0
        // Check match in shared pT
        float ptDiff = l1TkMuPt - tkRef->pt();
        float chi2Pt = ptDiff * ptDiff / (tkRef->ptError() * tkRef->ptError());
        if (chi2Pt < maxChi2_) {
          selectedTracksIndices.insert(tkIndex);
          outputTracks->push_back(*tkRef);
          std::cout << "MuonTracksSelectorFromL1TkMuon: Track with index " << tkIndex
                    << " matched to L1Tk muon with dR = " << sqrt(dR2) << " , chi2 pT = " << chi2Pt
                    << ", track z0 =  " << tkRef->dz() << ", L1TkMu z0 = " << l1TkMuZ0 << '\n';
        }
#endif
      } else {
        ++skippedDr;
      }
    }  // End loop over tracks
    std::cout << "MuonTracksSelectorFromL1TkMuon: Skipped " << skippedZ << " tracks due to dz cut\n"
              << " skipped " << skippedDr << " tracks due to dR cut\n skipped " << skippedPt
              << " tracks due to chi2Pt cut for L1Tk muon with index " << l1TkMuIndex << '\n';
  }  // End loop over L1Tk muons

  //std::cout << "MuonTracksSelectorFromL1TkMuon: Selected " << outputTracks->size() << " tracks out of "
  //          << tkCollectionH->size() << " input tracks, using " << l1TkMuCollectionH->size() << " L1 Tracker Muons\n";

  // Put output in the event
  iEvent.put(std::move(outputTracks));
}
#endif

void MuonTracksSelectorFromL1TkMuon::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("L1TkMuonInputCollection", edm::InputTag("l1tTkMuonsGmt"));
  // L1Tk muon selection parameters
  desc.add<double>("L1TkMuMinPt", 0);

  // Track selection parameters
  desc.add<edm::InputTag>("TrackInputCollection", edm::InputTag("generalTracks"));
  desc.add<double>("trackMinPt", 0.9);
  desc.add<double>("trackMaxEta", 3.0);
  desc.add<double>("maxChi2", 9);
  desc.add<double>("maxDr", 0.1);
  desc.add<double>("maxDz", 1.0);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonTracksSelectorFromL1TkMuon);