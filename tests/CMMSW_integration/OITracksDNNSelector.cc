/**
 * MuonOITracksDNNSelector.cc
 *
 * CMSSW EDProducer for DNN-based muon pixel track selection.
 * This module implements feature extraction matching the Python training exactly.
 *
 * Feature order (must match training):
 * ====================================
 * LOG FEATURES (log10(|x| + 1e-6)):
 *   0: track_p
 *   1: track_pt
 *   2: track_ptErr
 *   3: track_chi2
 *   4: track_normalizedChi2
 *   5: track_etaErr
 *   6: track_phiErr
 *   7: track_dszErr
 *   8: track_dxyErr
 *   9: track_dzErr
 *  10: track_qoverpErr
 *  11: track_lambdaErr
 *
 * PLAIN FEATURES (no transform):
 *  12: track_eta
 *  13: track_nPixelHits
 *  14: track_nTrkLays
 *  15: track_nFoundHits
 *  16: track_nLostHits
 *
 * DERIVED FEATURES:
 *  17: track_impact3D        (log10(dxy^2 + dz^2 + eps))
 *  18: track_impactSignificance (log10(sqrt((dxy/dxyErr)^2 + (dz/dzErr)^2) + eps))
 *  19: track_chi2PerHit      (log10(chi2 / max(nFoundHits, 1) + eps))
 *  20: track_hitEfficiency   (nFoundHits / max(nFoundHits + nLostHits, 1))
 *  21: track_sigmaPtOverPt   (log10(ptErr / pt + eps))
 *  22: track_relUncertaintyProduct (log10((ptErr/pt) * (qoverpErr/|qoverp|) + eps))
 *
 * Standalone Muon MATCHING FEATURES (if useStandaloneFeatures=true):
 *  23: Standalone_hasMatch     (1.0 if matched, 0.0 otherwise)
 *  24: Standalone_matchingScore
 *
 * LOW pT INDICATOR:
 *  25: is_low_pt           (1 / (1 + exp(clip((pt - 5.0) * 2.0, -20, 20))))
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include <cmath>
#include <algorithm>
#include <limits>

class MuonOITracksDNNSelector : public edm::stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>> {
public:
  explicit MuonOITracksDNNSelector(const edm::ParameterSet&, const cms::Ort::ONNXRuntime*);
  ~MuonOITracksDNNSelector() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  static std::unique_ptr<cms::Ort::ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const cms::Ort::ONNXRuntime*);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<float> extractFeatures(const reco::Track& track, const reco::TrackCollection& standaloneMuons) const;

  // Input tokens
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<reco::TrackCollection> standaloneMuonsToken_;

  // Model parameters
  const float decisionThreshold_;
  const bool useStandaloneMuonFeatures_;
  const int nFeatures_;

  // Matching parameters
  static constexpr float kMatchChi2 = 9.0f;  // 3 sigma
  static constexpr float kEpsilon = 1e-6f;

  // Imputation values for non-matched candidates
  static constexpr float kImputeMatchScore = 10.0f;
};

MuonOITracksDNNSelector::MuonOITracksDNNSelector(const edm::ParameterSet& iConfig, const cms::Ort::ONNXRuntime* cache)
    : tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      standaloneMuonsToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("standaloneMuons"))),
      decisionThreshold_(iConfig.getParameter<double>("decisionThreshold")),
      useStandaloneMuonFeatures_(iConfig.getParameter<bool>("useStandaloneMuonFeatures")),
      nFeatures_(iConfig.getParameter<int>("nFeatures")) {
  produces<reco::TrackCollection>();
  produces<std::vector<float>>("scores");
}

std::unique_ptr<cms::Ort::ONNXRuntime> MuonOITracksDNNSelector::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  edm::FileInPath modelPath(iConfig.getParameter<std::string>("modelPath"));
  return std::make_unique<cms::Ort::ONNXRuntime>(modelPath.fullPath());
}

void MuonOITracksDNNSelector::globalEndJob(const cms::Ort::ONNXRuntime* cache) {}

std::vector<float> MuonOITracksDNNSelector::extractFeatures(const reco::Track& track,
                                                            const reco::TrackCollection& standaloneMuons) const {
  std::vector<float> features;
  features.reserve(nFeatures_);

  const float p = track.p();
  const float pt = track.pt();
  const float ptErr = track.ptError();
  const float eta = track.eta();
  const float etaErr = track.etaError();
  const float phi = track.phi();
  const float phiErr = track.phiError();
  const float chi2 = track.chi2();
  const float normalizedChi2 = track.normalizedChi2();

  const float dszErr = track.dszError();
  const float dxy = track.dxy();
  const float dxyErr = track.dxyError();
  const float dz = track.dz();
  const float dzErr = track.dzError();
  const float qoverp = track.qoverp();
  const float qoverpErr = track.qoverpError();
  const float lambdaErr = track.lambdaError();

  const int nPixelHits = track.hitPattern().numberOfValidPixelHits();
  const int nTrkLays = track.hitPattern().trackerLayersWithMeasurement();
  const int nFoundHits = track.numberOfValidHits();
  const int nLostHits = track.numberOfLostHits();

  features.push_back(std::log10(std::abs(p) + kEpsilon));               // 0: p
  features.push_back(std::log10(std::abs(pt) + kEpsilon));              // 1: pt
  features.push_back(std::log10(std::abs(ptErr) + kEpsilon));           // 2: ptErr
  features.push_back(std::log10(std::abs(chi2) + kEpsilon));            // 3: chi2
  features.push_back(std::log10(std::abs(normalizedChi2) + kEpsilon));  // 4: normalizedChi2
  features.push_back(std::log10(std::abs(etaErr) + kEpsilon));          // 5: etaErr
  features.push_back(std::log10(std::abs(phiErr) + kEpsilon));          // 6: phiErr
  features.push_back(std::log10(std::abs(dszErr) + kEpsilon));          // 7: dszErr
  features.push_back(std::log10(std::abs(dxyErr) + kEpsilon));          // 8: dxyErr
  features.push_back(std::log10(std::abs(dzErr) + kEpsilon));           // 9: dzErr
  features.push_back(std::log10(std::abs(qoverpErr) + kEpsilon));       // 10: qoverpErr
  features.push_back(std::log10(std::abs(lambdaErr) + kEpsilon));       // 11: lambdaErr

  features.push_back(static_cast<float>(eta));         // 12: eta
  features.push_back(static_cast<float>(nPixelHits));  // 13: nPixelHits
  features.push_back(static_cast<float>(nTrkLays));    // 14: nTrkLays
  features.push_back(static_cast<float>(nFoundHits));  // 15: nFoundHits
  features.push_back(static_cast<float>(nLostHits));   // 16: nLostHits

  // 17: Impact Parameter 3D (log)
  const float impact3D = dxy * dxy + dz * dz;
  features.push_back(std::log10(impact3D + kEpsilon));

  // 18: Impact Significance (log)
  const float dxySignificance = dxy / std::max(dxyErr, kEpsilon);
  const float dzSignificance = dz / std::max(dzErr, kEpsilon);
  const float impactSignificance = std::sqrt(dxySignificance * dxySignificance + dzSignificance * dzSignificance);
  features.push_back(std::log10(impactSignificance + kEpsilon));

  // 19: Chi2 per hit (log)
  const float chi2PerHit = chi2 / std::max(nFoundHits, 1);
  features.push_back(std::log10(chi2PerHit + kEpsilon));

  // 20: Hit Efficiency (linear, NOT log)
  const float hitEfficiency = static_cast<float>(nFoundHits) / std::max(nFoundHits + nLostHits, 1);
  features.push_back(hitEfficiency);

  // 21: SigmaPt / Pt (log)
  const float sigmaPtOverPt = ptErr / std::max(pt, kEpsilon);
  features.push_back(std::log10(sigmaPtOverPt + kEpsilon));

  // 22: Relative Uncertainty Product (log)
  const float relUncertaintyProduct = sigmaPtOverPt * (qoverpErr / std::max(std::abs(qoverp), kEpsilon));
  features.push_back(std::log10(relUncertaintyProduct + kEpsilon));

  if (useStandaloneMuonFeatures_) {
    float bestScore = 25;  // 5 sigma
    // Loop over Standalone Muons and find the best match
    for (size_t standaloneMuonIndex = 0; standaloneMuonIndex != standaloneMuons.size(); ++standaloneMuonIndex) {
      auto muon = standaloneMuons[standaloneMuonIndex];
      float standaloneEta = muon.eta();
      float standaloneEtaErr = muon.etaError();
      float standalonePhi = muon.phi();
      float standalonePhiErr = muon.phiError();
      float standalonePt = muon.pt();
      float standalonePtErr = muon.ptError();
      float standaloneDz = muon.dz();
      float standaloneDzErr = muon.dzError();

      float chi2Eta = (eta - standaloneEta) * (eta - standaloneEta) /
                      (etaErr * etaErr + standaloneEtaErr * standaloneEtaErr + 1e-12f);
      float deltaPhi = reco::deltaPhi(phi, standalonePhi);
      float chi2Phi = deltaPhi * deltaPhi / (phiErr * phiErr + standalonePhiErr * standalonePhiErr + 1e-12f);
      float chi2Pt =
          (pt - standalonePt) * (pt - standalonePt) / (ptErr * ptErr + standalonePtErr * standalonePtErr + 1e-12f);
      float chi2Dz =
          (dz - standaloneDz) * (dz - standaloneDz) / (dzErr * dzErr + standaloneDzErr * standaloneDzErr + 1e-12f);

      float etaScore = chi2Eta / kMatchChi2;
      float phiScore = chi2Phi / kMatchChi2;
      float ptScore = chi2Pt / kMatchChi2;
      float dzScore = chi2Dz / kMatchChi2;

      float matchingScore = etaScore + phiScore + ptScore + dzScore;

      if (matchingScore < bestScore) {
        bestScore = matchingScore;
      }
    }  // End Standalone Muons Loop
    // Fill matching features
    if (bestScore < 25.0f) {
      features.push_back(1.0f);                              // 23: Standalone_hasMatch
      features.push_back(std::log10(bestScore + kEpsilon));  // 24: Standalone_matchingScore
    } else {
      features.push_back(0.0f);               // 23: Standalone_hasMatch
      features.push_back(kImputeMatchScore);  // 24: Standalone_matchingScore
    }
  }

  float exponent = (pt - 5.0f) * 2.0f;
  exponent = std::clamp(exponent, -20.0f, 20.0f);
  const float lowPtIndicator = 1.0f / (1.0f + std::exp(exponent));
  features.push_back(lowPtIndicator);

  return features;
}

void MuonOITracksDNNSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto selectedTracks = std::make_unique<reco::TrackCollection>();
  auto scores = std::make_unique<std::vector<float>>();

  // Get input collections
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tracksToken_, tracks);

  edm::Handle<reco::TrackCollection> standaloneMuons;
  iEvent.getByToken(standaloneMuonsToken_, standaloneMuons);

  if (!tracks.isValid() || tracks->empty()) {
    iEvent.put(std::move(selectedTracks));
    iEvent.put(std::move(scores), "scores");
    return;
  }

  // Prepare batch input
  std::vector<float> inputData;
  inputData.reserve(tracks->size() * nFeatures_);

  for (const auto& track : *tracks) {
    auto features = extractFeatures(track, *standaloneMuons);
    inputData.insert(inputData.end(), features.begin(), features.end());
  }

  // Run inference
  std::vector<std::vector<int64_t>> inputShapes = {
      {static_cast<int64_t>(tracks->size()), static_cast<int64_t>(nFeatures_)}};
  cms::Ort::FloatArrays inputTensor({inputData});

  auto outputs = globalCache()->run({"input"}, inputTensor, inputShapes, {"output"}, 1);

  // Model outputs probabilities directly (sigmoid is in the network)
  const auto& probs = outputs[0];

  //std::cout << "MuonOITracksDNNSelector - Processing " << tracks->size() << " tracks with threshold "
  //          << decisionThreshold_ << "\n";

  for (size_t i = 0; i < tracks->size(); ++i) {
    float prob = probs[i];
    //std::cout << "  Track " << i << ": DNN score = " << prob << "\n";

    // Clamp probability to valid range (safety check)
    prob = std::clamp(prob, 0.0f, 1.0f);

    scores->push_back(prob);

    if (prob >= decisionThreshold_) {
      //std::cout << "    -> Selected\n";
      selectedTracks->push_back((*tracks)[i]);
    }
  }

  //std::cout << "MuonOITracksDNNSelector - Selected " << selectedTracks->size() << " out of " << tracks->size()
  //          << " tracks\n";

  iEvent.put(std::move(selectedTracks));
  iEvent.put(std::move(scores), "scores");
}

void MuonOITracksDNNSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracks", edm::InputTag("hltPhase2L3OIMuCtfWithMaterialTracks"))
      ->setComment("Input track collection");
  desc.add<edm::InputTag>("standaloneMuons", edm::InputTag("hltL2MuonsFromL1TkMuon", "UpdatedAtVtx"))
      ->setComment("Standalone Muon collection for matching features");
  desc.add<std::string>("modelPath", "RecoMuon/L3TrackFinder/data/OI_track_selector.onnx")
      ->setComment("Path to ONNX model file (expects raw unscaled inputs, scaler fused)");
  desc.add<double>("decisionThreshold", 0.5)
      ->setComment("Probability threshold for track selection (use F2-optimal from training)");
  desc.add<bool>("useStandaloneMuonFeatures", true)->setComment("Include Standalone Muon matching features");
  desc.add<int>("nFeatures", 26)
      ->setComment("Total number of input features (17 base + 6 derived + 2 matching + 1 low_pt = 26)");

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonOITracksDNNSelector);