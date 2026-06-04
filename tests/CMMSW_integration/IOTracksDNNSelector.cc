/**
 * MuonIOTracksDNNSelector.cc
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
 * L1TkMuon MATCHING FEATURES (if useL1TkMuFeatures=true):
 *  23-26: Stub features (if available): nStubs, nStubs_Endcap, nStubs_Barrel, stubQual_max
 *  27: L1TkMu_hasMatch     (1.0 if matched, 0.0 otherwise)
 *  28: L1TkMu_dR2min       (log10(min_dR2 + eps), imputed with 0.1)
 *  29: L1TkMu_dPtNorm      (log10(|pt - l1_pt|/l1_pt + eps), imputed with 1.0)
 *  30: L1TkMu_chi2Pt       (log10((pt - l1_pt)^2 / ptErr^2 + eps), imputed with 10.0)
 *  31: L1TkMu_matchingScore (log10(min_dR2 * (1 + dPtNorm) + eps), imputed with 0.2)
 *
 * LOW pT INDICATOR:
 *  32: is_low_pt           (1 / (1 + exp(clip((pt - 5.0) * 2.0, -20, 20))))
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
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include <cmath>
#include <algorithm>
#include <limits>

class MuonIOTracksDNNSelector : public edm::stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>> {
public:
  explicit MuonIOTracksDNNSelector(const edm::ParameterSet&, const cms::Ort::ONNXRuntime*);
  ~MuonIOTracksDNNSelector() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  static std::unique_ptr<cms::Ort::ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const cms::Ort::ONNXRuntime*);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<float> extractFeatures(const reco::Track& track, const l1t::TrackerMuonCollection& l1TkMuons) const;

  // Input tokens
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<l1t::TrackerMuonCollection> l1TkMuonsToken_;

  // Model parameters
  const float decisionThreshold_;
  const bool useL1TkMuFeatures_;
  const bool useStubFeatures_;
  const int nFeatures_;

  // L1 Matching parameters
  static constexpr float kMatchDR2Cut = 0.09f;    // 0.3^2
  static constexpr float kMatchChi2PtCut = 9.0f;  // 3 sigma
  static constexpr float kEpsilon = 1e-6f;

  // Imputation values for non-matched L1 features
  static constexpr float kImputeDR2 = 0.1f;
  static constexpr float kImputeDPtNorm = 1.0f;
  static constexpr float kImputeChi2Pt = 10.0f;
  static constexpr float kImputeMatchScore = 0.2f;
};

MuonIOTracksDNNSelector::MuonIOTracksDNNSelector(const edm::ParameterSet& iConfig, const cms::Ort::ONNXRuntime* cache)
    : tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      l1TkMuonsToken_(consumes<l1t::TrackerMuonCollection>(iConfig.getParameter<edm::InputTag>("l1TkMuons"))),
      decisionThreshold_(iConfig.getParameter<double>("decisionThreshold")),
      useL1TkMuFeatures_(iConfig.getParameter<bool>("useL1TkMuFeatures")),
      useStubFeatures_(iConfig.getParameter<bool>("useStubFeatures")),
      nFeatures_(iConfig.getParameter<int>("nFeatures")) {
  produces<reco::TrackCollection>();
  produces<std::vector<float>>("scores");
}

std::unique_ptr<cms::Ort::ONNXRuntime> MuonIOTracksDNNSelector::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  edm::FileInPath modelPath(iConfig.getParameter<std::string>("modelPath"));
  return std::make_unique<cms::Ort::ONNXRuntime>(modelPath.fullPath());
}

void MuonIOTracksDNNSelector::globalEndJob(const cms::Ort::ONNXRuntime* cache) {}

std::vector<float> MuonIOTracksDNNSelector::extractFeatures(const reco::Track& track,
                                                            const l1t::TrackerMuonCollection& l1TkMuons) const {
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

  if (useL1TkMuFeatures_) {
    // Find best L1 match (minimum deltaR2)
    float minDR2 = std::numeric_limits<float>::max();
    float matchedL1Pt = -1.0f;
    bool hasL1Match = false;
    int bestIndex = -1;
    for (size_t l1TkMuIndex = 0; l1TkMuIndex != l1TkMuons.size(); ++l1TkMuIndex) {
      auto l1TkMu = l1TkMuons.at(l1TkMuIndex);
      auto l1TTrack = l1TkMu.trkPtr();
      if (!l1TTrack)
        continue;

      const float l1Eta = l1TTrack->momentum().eta();
      const float l1Phi = l1TTrack->momentum().phi();
      const float l1Pt = l1TTrack->momentum().perp();

      float chi2pT = (pt - l1Pt) * (pt - l1Pt) / (ptErr * ptErr);
      if (chi2pT > 9.0)
        continue;
      const float dR2 = reco::deltaR2(eta, phi, l1Eta, l1Phi);

      if (dR2 < minDR2) {
        minDR2 = dR2;
        matchedL1Pt = l1Pt;
        bestIndex = l1TkMuIndex;
      }
    }

    // Compute matching quantities for best match
    float dPtNorm = kImputeDPtNorm;
    float chi2Pt = kImputeChi2Pt;
    float matchingScore = kImputeMatchScore;

    if (matchedL1Pt > 0) {
      dPtNorm = std::abs(pt - matchedL1Pt) / matchedL1Pt;

      const float ptDiff = pt - matchedL1Pt;
      chi2Pt = (ptDiff * ptDiff) / (ptErr * ptErr);

      matchingScore = minDR2 * (1.0f + dPtNorm);
    }

    hasL1Match = (minDR2 < kMatchDR2Cut) && (chi2Pt < kMatchChi2PtCut) && (matchedL1Pt > 0);

    // Add stub features if enabled
    if (useStubFeatures_) {
      std::optional<l1t::TrackerMuon> bestL1 =
          bestIndex != -1 ? std::make_optional(l1TkMuons[bestIndex]) : std::nullopt;
      features.push_back(bestL1 ? bestL1->stubs().size() : 0);  // 23: L1TkMu_nStubs
      int nStubsEndcap = 0;
      int nStubsBarrel = 0;
      int maxStubQuality = 0;
      int minDepthRegion = std::numeric_limits<int>::max();
      int bestStubIndex = -1;
      if (bestL1) {
        for (size_t stubIndex = 0; stubIndex != bestL1->stubs().size(); ++stubIndex) {
          auto stubRef = bestL1->stubs().at(stubIndex);
          if (stubRef.isNull())
            continue;
          if (stubRef->type() == 0)
            nStubsEndcap += 1;
          if (stubRef->type() == 1)
            nStubsBarrel += 1;
          if (stubRef->quality() > maxStubQuality ||
              (stubRef->quality() == maxStubQuality && stubRef->depthRegion() < minDepthRegion)) {
            minDepthRegion = stubRef->depthRegion();
            maxStubQuality = stubRef->quality();
            bestStubIndex = stubIndex;
          }
        }
      }
      auto bestStub = bestStubIndex != -1 ? std::make_optional(bestL1->stubs().at(bestStubIndex)) : std::nullopt;
      features.push_back(bestStub ? nStubsEndcap : 0);                 // 24: L1TkMu_nStubs_Endcap
      features.push_back(bestStub ? nStubsBarrel : 0);                 // 25: L1TkMu_nStubs_Barrel
      features.push_back(bestStub ? (*bestStub)->quality() : 0);       // 26: L1TkMu_stubQual_max
      features.push_back(bestStub ? (*bestStub)->etaRegion() : -1);    // 27: L1TkMu_stubEtaRegion
      features.push_back(bestStub ? (*bestStub)->phiRegion() : -1);    // 28: L1TkMu_stubPhiRegion
      features.push_back(bestStub ? (*bestStub)->depthRegion() : -1);  // 29: L1TkMu_stubDepthRegion
    }

    // Feature: L1TkMu_hasMatch (binary)
    features.push_back(hasL1Match ? 1.0f : 0.0f);

    // For non-matched tracks, use imputation values (matching Python impute_and_log behavior)
    if (hasL1Match) {
      // Feature: L1TkMu_dR2min (log)
      features.push_back(std::log10(std::abs(minDR2) + kEpsilon));

      // Feature: L1TkMu_dPtNorm (log)
      features.push_back(std::log10(std::abs(dPtNorm) + kEpsilon));

      // Feature: L1TkMu_chi2Pt (log)
      features.push_back(std::log10(std::abs(chi2Pt) + kEpsilon));

      // Feature: L1TkMu_matchingScore (log)
      features.push_back(std::log10(std::abs(matchingScore) + kEpsilon));
    } else {
      // Imputed values (from Python fill_value parameters)
      features.push_back(std::log10(std::abs(kImputeDR2) + kEpsilon));         
      features.push_back(std::log10(std::abs(kImputeDPtNorm) + kEpsilon));     
      features.push_back(std::log10(std::abs(kImputeChi2Pt) + kEpsilon));      
      features.push_back(std::log10(std::abs(kImputeMatchScore) + kEpsilon)); 
    }
  }

  float exponent = (pt - 5.0f) * 2.0f;
  exponent = std::clamp(exponent, -20.0f, 20.0f);
  const float lowPtIndicator = 1.0f / (1.0f + std::exp(exponent));
  features.push_back(lowPtIndicator);

  return features;
}

void MuonIOTracksDNNSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto selectedTracks = std::make_unique<reco::TrackCollection>();
  auto scores = std::make_unique<std::vector<float>>();

  // Get input collections
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tracksToken_, tracks);

  edm::Handle<l1t::TrackerMuonCollection> l1TkMuons;
  iEvent.getByToken(l1TkMuonsToken_, l1TkMuons);

  if (!tracks.isValid() || tracks->empty()) {
    iEvent.put(std::move(selectedTracks));
    iEvent.put(std::move(scores), "scores");
    return;
  }

  // Prepare batch input
  std::vector<float> inputData;
  inputData.reserve(tracks->size() * nFeatures_);

  for (const auto& track : *tracks) {
    auto features = extractFeatures(track, *l1TkMuons);
    inputData.insert(inputData.end(), features.begin(), features.end());
  }

  // Run inference
  std::vector<std::vector<int64_t>> inputShapes = {
      {static_cast<int64_t>(tracks->size()), static_cast<int64_t>(nFeatures_)}};
  cms::Ort::FloatArrays inputTensor({inputData});

  auto outputs = globalCache()->run({"input"}, inputTensor, inputShapes, {"output"}, 1);

  // Model outputs probabilities directly (sigmoid is in the network)
  const auto& probs = outputs[0];

  //std::cout << "MuonIOTracksDNNSelector - Processing " << tracks->size() << " tracks with threshold "
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

  //std::cout << "MuonIOTracksDNNSelector - Selected " << selectedTracks->size() << " out of " << tracks->size()
  //          << " tracks\n";

  iEvent.put(std::move(selectedTracks));
  iEvent.put(std::move(scores), "scores");
}

void MuonIOTracksDNNSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("tracks", edm::InputTag("hltPhase2L3FromL1TkMuonPixelTracks"))
      ->setComment("Input track collection");
  desc.add<edm::InputTag>("l1TkMuons", edm::InputTag("l1tTkMuonsGmt"))
      ->setComment("L1 Tracker Muon collection for matching features");
  desc.add<std::string>("modelPath", "RecoMuon/L3TrackFinder/data/pixel_track_selector.onnx")
      ->setComment("Path to ONNX model file (expects raw unscaled inputs, scaler fused)");
  desc.add<double>("decisionThreshold", 0.5)
      ->setComment("Probability threshold for track selection (use F2-optimal from training)");
  desc.add<bool>("useL1TkMuFeatures", true)->setComment("Include L1 Tracker Muon matching features");
  desc.add<bool>("useStubFeatures", true)->setComment("Include stub-related features (requires stub info in event)");
  desc.add<int>("nFeatures", 36)
      ->setComment("Total number of input features (17 base + 6 derived + 6 L1 + 4 stub = 33)");

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonIOTracksDNNSelector);