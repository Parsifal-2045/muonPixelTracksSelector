# Selector for muon tracks in the Pixel detector of CMS
This repository contains code for the exploration and development of a ML-based high purity selection for muon tracks in the Pixel detector of the CMS experiment. The project aims to be deployed for the Phase-2 High Level Trigger reconstruction, taking advantage of the improved PixelTracks reconstruction to effectively track muons without running a separate dedicated tracking iteration.
Development base on CA extension for Phase 2 and previous integration of Alpaka-based PixelTracks in muons sequences, as shown in [22/07 HLT Upgrade meeting](https://indico.cern.ch/event/1570468/#4-single-iteration-io-muon-rec). NB the results shown were preliminary, further developments used as baseline for this work: CA extension implemented in CMSSW (PR) and PixelTracks usage for muon tracking (PR).

This work aims to reduce the fake tracks (i.e. not matched to a signal muon as per validation selection) present at the tracking level for HLT muon reconstruction, with the aim to replace the cut-based MVA selection typically applied to general tracks (meaning PixelTracks extended to the Outer Tracker via ckf). 
Development steps:

- [x] Extend CMSSW ntuple producer to include MC truth information (using the same simToReco and recoToSim associators used by the standard validation sequences) ([commit](https://github.com/Parsifal-2045/cmssw/commit/04dd7388b7e0e2b6c0dc053f85071fb3d856b857) on private CMSSW branch)
- [x] Extract ntuples with variables of interest (mostly track quality parameters) and MC truth information (e.g. matched, duplicate, simulated pT, ...) ([commit](https://github.com/Parsifal-2045/cmssw/commit/f42d8e5b6ccfaa854c9a196621dfc4d7140c86e3) on private CMSSW branch)
- [x] Visual inspection of extracted variables to figure out which ones could be used to discriminate good tracks from fakes ([link](https://lferragi.web.cern.ch/plots/muon_hlt_phase2/muonPixelTracksSelector/) to plots and ntuples)
- [ ] Implementation and training of a ML model (probably BDT/DNN) on a varied sample of events with PU up to 200 (Phase-2 conditions)
- [ ] Analysis of model's result and fine tuning
- [ ] If the results are good, export the model to ONNX for future inclusion in CMSSW

The first three steps (marked as completed here) have been completed on a sample of 10'000 ZMM events at 200 PU. This is not representative of the final sample to be used for the training of the model, it's just a bootstrap start to familiarise with the variables and take a look at some rough cuts.
