#!/usr/bin/env python3
"""
Lund Jet Plane Reweighting Example Script for generic Ntuple/NanoAOD.
This script requires CMSSW environment to be set up. Execute `scramv1 runtime -sh`
before running this script.
This script demonstrates an abstracted, modular pipeline to apply Lund Jet Plane
reweighting, calculate substructure cut efficiencies, and derive Scale Factors (SF)
with their associated uncertainties.
"""

import sys
import os
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional
import uuid
import shutil
import concurrent.futures
import importlib

import numpy as np
import awkward as ak
import ROOT

# Assuming the script is run from within the 'scripts' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(os.path.join(current_dir, "../"))

try:
    from utils.Utils import LundReweighter
except ImportError:
    print("Error: Could not import LundReweighter. Ensure you are running from the scripts/ directory.")
    sys.exit(1)

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Event, InputTree

# ==============================================================================
# Helper Functions
# ==============================================================================

def ang_dist(phi1: float, phi2: float) -> float:
    """Calculates the angular distance between two azimuthal angles."""
    dphi = phi1 - phi2
    if dphi < -math.pi:
        dphi += 2.0 * math.pi
    if dphi > math.pi:
        dphi -= 2.0 * math.pi
    return dphi

def deltaR(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """Calculates the delta R distance in the eta-phi plane."""
    return math.sqrt((eta1 - eta2)**2 + ang_dist(phi1, phi2)**2)

def parse_part(p: Any) -> List[float]:
    """Extract Gen Quark features"""
    # Use 999.0 as a safety value for eta/phi to avoid false matching at the detector center
    return [p.pt, p.eta, p.phi, float(p.pdgId)] if p else [0.0, 999.0, 999.0, 0.0]

# ==============================================================================
# 1. Matching Logic
# ==============================================================================

def get_matched_quarks(jet: Any, quarks: List[Any], dr_cut: float = 0.8) -> List[Any]:
    """
    Finds all quarks within the jet radius.
    """
    matched = []
    for q in quarks:
        if q is not None:
            if deltaR(jet.eta, jet.phi, q.eta, q.phi) < dr_cut:
                matched.append(q)
    return matched

# ==============================================================================
# 2. Generator Level Extraction
# ==============================================================================

def extract_gen_particles(event: Event, genpart_branch: str = "GenPart", verbose: bool = False) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """
    Parses the generator-level particles to identify quarks defining the prongs.
    This implementation targets a ttbar topology.

    Args:
        event (Event): The current NanoAOD event.
        genpart_branch (str): Name of the branch containing Gen particles.
        verbose (bool): If True, prints extra debugging information.

    Returns:
        Tuple: Contains 10 generator objects in the following order:
               (top, anti_top, W, anti_W, q1a, q1b, b1, q2a, q2b, b2)
               If a particle is not found, its corresponding slot is None.
    """
    def isFinal(genPart: Any) -> bool:
        """Check if isLastCopy flag is set (Pythia)."""
        mask = 1 << 13  # 13th bit of status flag
        return (genPart.statusFlags & mask) != 0

    top_ID, W_ID, B_ID = 6, 24, 5
    MAXLEP_ID = 16

    GenPartsColl = Collection(event, genpart_branch)

    top = anti_top = W = anti_W = None
    q1a = q1b = b1 = q2a = q2b = b2 = None

    # First pass: Find top and W bosons
    for genPart in GenPartsColl:
        if abs(genPart.pdgId) == top_ID and isFinal(genPart):
            if genPart.pdgId > 0: top = genPart if top is None else top
            else: anti_top = genPart if anti_top is None else anti_top

        if abs(genPart.pdgId) == W_ID and isFinal(genPart):
            if genPart.pdgId > 0: W = genPart if W is None else W
            else: anti_W = genPart if anti_W is None else anti_W

    # Second pass: Find quarks from W and top decays
    for genPart in GenPartsColl:
        m = genPart.genPartIdxMother
        mother = GenPartsColl[m] if m >= 0 else None

        # Quarks/leptons from W decay (handles ttbar, tW, WW)
        if abs(genPart.pdgId) <= MAXLEP_ID and m >= 0:
            if mother is W:
                if genPart.pdgId > 0: q1a = genPart if q1a is None else q1a
                else: q1b = genPart if q1b is None else q1b
            elif mother is anti_W:
                if genPart.pdgId > 0: q2a = genPart if q2a is None else q2a
                else: q2b = genPart if q2b is None else q2b

        # b quarks from top decay
        if abs(genPart.pdgId) == B_ID and m >= 0:
            if mother is top: b1 = genPart if b1 is None else b1
            elif mother is anti_top: b2 = genPart if b2 is None else b2

    return (top, anti_top, W, anti_W, q1a, q1b, b1, q2a, q2b, b2)

# ==============================================================================
# 3. Selection
# ==============================================================================

def apply_selections(inTree: InputTree, triggers: Optional[List[str]] = None) -> bool:
    """
    Applies event-level preselection (e.g., passing trigger paths).

    Args:
        inTree (InputTree): The input ROOT tree interface.
        triggers (List[str]): List of triggers to check (logical OR).

    Returns:
        bool: True if the event passes selections, False otherwise.
    """
    if not triggers:
        return True

    pass_trigger = False
    for trig in triggers:
        # Using logical OR across triggers
        pass_trigger = pass_trigger or inTree.readBranch(trig)

    return pass_trigger

# ==============================================================================
# 4. Input Processing
# ==============================================================================

def process_inputs(
    input_file: Any,
    fatjet_branch: str = "FatJet",
    pfcand_branch: str = "PFCand",
    genpart_branch: str = "GenPart",
    fatjet_pfcand_branch: str = "FatJetPFCand",
    genweight_branch: str = "genWeight",
    features: List[str] = ["tau1", "tau2"],
    max_process_jets: int = 5000,
    jet_min_pt: float = 400.0,
    triggers: Optional[List[str]] = None,
    topology: str = "b2b"
) -> Tuple[np.ndarray, np.ndarray, List[List[List[float]]], np.ndarray, np.ndarray]:
    """
    Processes the input ROOT tree, applying selections and extracting arrays
    for jets, gen quarks, and PF candidates.

    Args:
        input_file (ROOT.TFile): Opened ROOT file.
        fatjet_branch (str): Branch name for FatJets.
        pfcand_branch (str): Branch name for PFCands.
        genpart_branch (str): Branch name for GenParts.
        fatjet_pfcand_branch (str): Branch name linking FatJets and PFCands.
        max_process_jets (int): Max jets to extract per file. Use a negative value to process all.
        jet_min_pt (float): Minimum pT threshold for the target FatJet.
        triggers (List[str], optional): Triggers for preselection.

    Returns:
        Tuple:
            - jets (np.ndarray): Shape (N, 6) Array of selected jet properties.
            - quarks (np.ndarray): Shape (N, 6, 4) Array of gen-level quarks.
            - cands (List): N-length list of PF candidates associated with jets.
    """
    TTree = input_file.Get("Events")
    if not TTree:
        raise ValueError("Could not locate 'Events' tree in input file.")

    inTree = InputTree(TTree)

    n_selected_jets = 0
    n_events_evaluated = 0
    n_jets_per_event = []
    jets_list = []
    quarks_list = []
    cands_list = []
    gen_weights_list = []

    for entry in range(inTree.entries):
        event = Event(inTree, entry)

        if not apply_selections(inTree, triggers):
            continue

        # Get Event Weight
        try:
            event_weight = inTree.readBranch(genweight_branch)
        except RuntimeError:
            event_weight = 1.0

        AK8Jets = Collection(event, fatjet_branch)
        PFCands = Collection(event, pfcand_branch)
        FatJetPFCands = Collection(event, fatjet_pfcand_branch)

        if len(AK8Jets) == 0 or AK8Jets[0].pt < jet_min_pt:
            continue

        # Extract gen level topology
        gen_parts = extract_gen_particles(event, genpart_branch=genpart_branch)
        _, _, _, _, q1a, q1b, b1, q2a, q2b, b2 = gen_parts

        # Pool of potential quarks for matching
        potential_quarks_evt = [q1a, q1b, q2a, q2b] # Copy for the event
        source1_active = True
        source2_active = True

        n_jets_in_event = 0
        matched_jets_in_event = []
        matched_quarks_in_event = []

        for i, jet in enumerate(AK8Jets):
            # Smart indexing: Use existing 'idx' if present (e.g. from pre-filtered Ntuples),
            # otherwise use the current collection index.
            if "idx" not in vars(jet):
                try:
                    _ = jet.idx
                except Exception:
                    jet.idx = i

            if jet.pt > jet_min_pt and abs(jet.eta) < 2.4:
                # Selection based on topology
                if topology == "boost":
                    if len(potential_quarks_evt) >= 2:
                        # Boosted: allow matching from the full pool (e.g. H->4q, WW-merged)
                        matched_q = get_matched_quarks(jet, potential_quarks_evt)
                        if len(matched_q) >= 2:
                            matched_jets_in_event.append(jet)
                            matched_quarks_in_event.append(matched_q)
                            # Remove matched quarks from the pool to avoid double counting
                            for q in matched_q:
                                if q in potential_quarks_evt:
                                    potential_quarks_evt.remove(q)
                    else:
                        break # No more jets can possibly match at least 2 quarks
                else:
                    # b2b (default): separate Top1 and Top2 decay chains (e.g. ttbar, tW)
                    matched_q1 = get_matched_quarks(jet, [q1a, q1b, b1]) if source1_active else []
                    matched_q2 = get_matched_quarks(jet, [q2a, q2b, b2]) if source2_active else []

                    if len(matched_q1) >= 2:
                        matched_jets_in_event.append(jet)
                        matched_quarks_in_event.append(matched_q1)
                        source1_active = False
                    elif len(matched_q2) >= 2:
                        matched_jets_in_event.append(jet)
                        matched_quarks_in_event.append(matched_q2)
                        source2_active = False

                    if not source1_active and not source2_active:
                        break # Both decay chains consumed

        n_events_evaluated += 1
        n_jets_per_event.append(len(matched_jets_in_event))

        if len(matched_jets_in_event) == 0:
            continue

        for jet, matched_q in zip(matched_jets_in_event, matched_quarks_in_event):
            jet_data = [
                jet.pt, jet.eta, jet.phi, getattr(jet, "msoftdrop", 0)
            ]
            for feat in features:
                jet_data.append(getattr(jet, feat, 0))

            jets_list.append(jet_data)

            # The full 6 slots standard for the output
            full_quarks = [q1a, q1b, b1, q2a, q2b, b2]
            quark_group = []
            for q in full_quarks:
                if q is not None and q in matched_q:
                    quark_group.append(parse_part(q))
                else:
                    quark_group.append([0.0, 999.0, 999.0, 0.0])
            quarks_list.append(quark_group)

            # Extract PF Candidates for the selected jet
            jet_PFCands = []
            for cand_map in FatJetPFCands:
                if cand_map.jetIdx == jet.idx:
                    idx = cand_map.pfCandIdx
                    cand_obj = PFCands[idx]
                    cand_vec = ROOT.Math.PtEtaPhiMVector(
                        cand_obj.pt, cand_obj.eta, cand_obj.phi, getattr(cand_obj, "mass", 0)
                    )
                    jet_PFCands.append([cand_vec.Px(), cand_vec.Py(), cand_vec.Pz(), cand_vec.E()])

            cands_list.append(jet_PFCands)
            gen_weights_list.append(event_weight)
            n_selected_jets += 1

        if max_process_jets >= 0 and n_selected_jets >= max_process_jets:
            break

    return np.array(jets_list), np.array(quarks_list), cands_list, np.array(n_jets_per_event), np.array(gen_weights_list)

## ==============================================================================
# 5. Global Distortion (Pass 1)
# ==============================================================================

def worker_pass1(fpath: str, ratio_file_path: str, args: Any, triggers: List[str]) -> Optional[np.ndarray]:
    """
    Worker for Pass 1: creates a local copy of ratio.root, initializes LundReweighter,
    computes the local distortion histogram, and returns the bin contents as a numpy array.
    """
    worker_id = str(uuid.uuid4())
    local_ratio = f"{ratio_file_path}.{worker_id}.tmp"
    shutil.copy2(ratio_file_path, local_ratio)

    try:
        f_ratio = ROOT.TFile.Open(local_ratio)
        if not f_ratio or f_ratio.IsZombie():
            return None

        LP_rw = LundReweighter(f_ratio=f_ratio, use_CA=False, pf_pt_min=1.0)
        h_lp_signal = LP_rw.h_mc.Clone(f"h_lp_signal_{worker_id}")
        h_lp_signal.Reset()

        f_in = ROOT.TFile.Open(fpath)
        if not f_in or f_in.IsZombie():
            f_ratio.Close()
            return None

        jets, quarks, cands, n_jets_per_event, _ = process_inputs(
            input_file=f_in,
            fatjet_branch=args.fatjet,
            pfcand_branch=args.pfcand,
            genpart_branch=args.genpart,
            fatjet_pfcand_branch=args.fatjet_pfcand,
            genweight_branch=args.genweight,
            features=args.features,
            max_process_jets=args.max_process_jets,
            jet_min_pt=args.min_pt,
            triggers=triggers,
            topology=args.topology
        )
        f_in.Close()

        if len(jets) > 0:
            ak8_jets = jets[:, :4]
            gen_parts_eta_phi = quarks[:, :, 1:3]
            for j in range(len(jets)):
                reclust_nom, _, _ = LP_rw.get_splittings_and_matching(cands[j], gen_parts_eta_phi[j], ak8_jets[j])
                if not reclust_nom.badmatch:
                    LP_rw.fill_lund_plane(h_lp_signal, reclust_obj=reclust_nom)

        ncells = h_lp_signal.GetNcells()
        contents = np.zeros(ncells, dtype=np.float64)
        for i in range(ncells):
            contents[i] = h_lp_signal.GetBinContent(i)

        f_ratio.Close()
        return contents

    except Exception as e:
        print(f"Error in worker_pass1 for {fpath}: {e}")
        return None
    finally:
        if os.path.exists(local_ratio):
            os.remove(local_ratio)

def get_global_distortion(inputs: List[str], LP_rw_global: Any, args: Any, triggers: List[str]) -> Any:
    """
    Pass 1: Reads all files in parallel to compute the global Lund Plane signal distribution
    and calculates the global distortion ratio.
    """
    print(f"\n[Pass 1] Computing global Lund Plane distribution for distortion systematic (using {args.workers} workers)...")
    h_lp_signal_global = LP_rw_global.h_mc.Clone("h_lp_signal_global")
    h_lp_signal_global.Reset()

    ncells = h_lp_signal_global.GetNcells()
    global_contents = np.zeros(ncells, dtype=np.float64)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_pass1, fpath, args.ratio, args, triggers): fpath for fpath in inputs}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            fpath = futures[future]
            print(f"  -> Pass 1 finished for {fpath} ({i+1}/{len(inputs)})")
            res = future.result()
            if res is not None:
                global_contents += res

    # Reconstruct the global TH3D
    for i in range(ncells):
        h_lp_signal_global.SetBinContent(i, global_contents[i])
        # Sumw2 is implicitly handled for weight=1 by sqrt(content), but let's be explicit
        h_lp_signal_global.GetSumw2().SetAt(global_contents[i], i)

    h_dummy = LP_rw_global.h_mc.Clone("h_dummy")
    h_dummy.Reset()
    h_distortion_ratio = LP_rw_global.make_LP_ratio(LP_rw_global.h_mc, h_dummy, h_lp_signal_global)

    from utils.LundReweighter import cleanup_ratio
    cleanup_ratio(h_distortion_ratio, h_min=0.2, h_max=5.0)

    return h_distortion_ratio

# ==============================================================================
# 6. Raw Weight Calculation (Pass 2)
# ==============================================================================

def calculate_raw_weights(
    jets: np.ndarray,
    quarks: np.ndarray,
    cands: List[List[List[float]]],
    LP_rw: Any,
    h_distortion_ratio: Any,
    rand_noise: np.ndarray,
    pt_rand_noise: np.ndarray,
    chunk_size: int = 5000,
    w_max: float = 10.0
) -> Dict[str, Any]:
    """
    Computes the raw (unnormalized) Lund Plane weights for a single file in chunks.
    """
    ak8_jets = jets[:, :4]
    gen_parts_eta_phi = quarks[:, :, 1:3]
    gen_parts_pdg_id = quarks[:, :, 3]

    num_jets = len(jets)
    num_chunks = math.ceil(num_jets / chunk_size)
    nToys = rand_noise.shape[0]

    LP_weights_combined = {}

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_jets)

        chunk_cands = cands[start:end]
        chunk_eta_phi = gen_parts_eta_phi[start:end]
        chunk_jets = ak8_jets[start:end]
        chunk_pdg_id = gen_parts_pdg_id[start:end]

        chunk_weights = LP_rw.get_all_weights(
            chunk_cands, chunk_eta_phi, chunk_jets, gen_parts_pdg_ids=chunk_pdg_id,
            nToys=nToys, rand_noise=rand_noise, pt_rand_noise=pt_rand_noise,
            normalize=False, distortion_sys=False
        )

        dist_up = np.zeros(len(chunk_jets))
        dist_down = np.zeros(len(chunk_jets))
        for j in range(len(chunk_jets)):
            reclust_nom, _, _ = LP_rw.get_splittings_and_matching(chunk_cands[j], chunk_eta_phi[j], chunk_jets[j])
            distortion_weight, _, _ = LP_rw.reweight_lund_plane(h_rw=h_distortion_ratio, reclust_obj=reclust_nom, sys_str='distortion')

            dist_up[j] = chunk_weights['nom'][j] * distortion_weight
            if distortion_weight > 0:
                dist_down[j] = chunk_weights['nom'][j] / distortion_weight
            else:
                dist_down[j] = chunk_weights['nom'][j]

        chunk_weights['distortion_up'] = dist_up
        chunk_weights['distortion_down'] = dist_down

        if not LP_weights_combined:
            for key, val in chunk_weights.items():
                if isinstance(val, np.ndarray):
                    LP_weights_combined[key] = [val]
                elif isinstance(val, list):
                    LP_weights_combined[key] = list(val)
                else:
                    LP_weights_combined[key] = val
        else:
            for key, val in chunk_weights.items():
                if isinstance(val, np.ndarray):
                    LP_weights_combined[key].append(val)
                elif isinstance(val, list):
                    LP_weights_combined[key].extend(val)

    for key, val in LP_weights_combined.items():
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], np.ndarray):
            LP_weights_combined[key] = np.concatenate(val, axis=0)

        # Apply clipping to all weight-related arrays
        if isinstance(LP_weights_combined[key], np.ndarray) and any(x in key for x in ['nom', 'up', 'down', 'vars']):
            LP_weights_combined[key] = np.clip(LP_weights_combined[key], 0.0, w_max)

    return LP_weights_combined

def worker_pass2(fpath: str, ratio_file_path: str, args: Any, triggers: List[str],
                 rand_noise: np.ndarray, pt_rand_noise: np.ndarray,
                 dist_contents: np.ndarray) -> Tuple[str, Dict[str, Any], np.ndarray]:
    """
    Worker for Pass 2: processes one file, reconstructs global distortion ratio,
    and calculates raw weights.
    """
    worker_id = str(uuid.uuid4())
    local_ratio = f"{ratio_file_path}.{worker_id}.tmp"
    shutil.copy2(ratio_file_path, local_ratio)

    try:
        f_ratio = ROOT.TFile.Open(local_ratio)
        if not f_ratio or f_ratio.IsZombie():
            return fpath, {}, np.array([]), np.array([])

        LP_rw = LundReweighter(f_ratio=f_ratio, use_CA=False, pf_pt_min=1.0)

        # Reconstruct h_distortion_ratio from contents
        h_distortion_ratio = LP_rw.h_mc.Clone(f"h_dist_{worker_id}")
        h_distortion_ratio.Reset()
        for i in range(h_distortion_ratio.GetNcells()):
            h_distortion_ratio.SetBinContent(i, dist_contents[i])

        f_in = ROOT.TFile.Open(fpath)
        if not f_in or f_in.IsZombie():
            f_ratio.Close()
            return fpath, {}, np.array([]), np.array([])

        jets, quarks, cands, n_jets_per_event, gen_weights = process_inputs(
            input_file=f_in,
            fatjet_branch=args.fatjet,
            pfcand_branch=args.pfcand,
            genpart_branch=args.genpart,
            fatjet_pfcand_branch=args.fatjet_pfcand,
            genweight_branch=args.genweight,
            features=args.features,
            max_process_jets=args.max_process_jets,
            jet_min_pt=args.min_pt,
            triggers=triggers,
            topology=args.topology
        )
        f_in.Close()

        if len(jets) == 0:
            f_ratio.Close()
            return fpath, {}, np.array([]), np.array([])

        raw_weights = calculate_raw_weights(jets, quarks, cands, LP_rw, h_distortion_ratio, rand_noise, pt_rand_noise, args.chunk_size, args.w_max)
        f_ratio.Close()

        return fpath, raw_weights, jets, n_jets_per_event, gen_weights
    except Exception as e:
        print(f"Error in worker_pass2 for {fpath}: {e}")
        return fpath, {}, np.array([]), np.array([]), np.array([])
    finally:
        if os.path.exists(local_ratio):
            os.remove(local_ratio)

# ==============================================================================
# 7. Global Normalization (Pass 3)
# ==============================================================================

def unflatten_event_weights(jet_weights: np.ndarray, n_jets_per_event: np.ndarray) -> np.ndarray:
    if len(jet_weights) == 0 or len(n_jets_per_event) == 0:
        return np.array([])
    # Create jagged array and compute product over jets in each event
    j_weights = ak.unflatten(jet_weights, n_jets_per_event)
    evt_weights = ak.prod(j_weights, axis=1)
    return ak.to_numpy(evt_weights)

def get_normalized_event_weights(all_raw_weights: Dict[str, Dict[str, Any]], all_n_jets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Concatenates weights from all files, constructs EVENT-LEVEL weights,
    and normalizes them such that the mean weight of events with >= 1 jet is 1.0.
    Returns a dictionary of normalized event weights keyed by filename.
    """
    print("\n[Pass 3] Calculating normalized event-level weights for MC calibration...")
    fnames = list(all_raw_weights.keys())
    if not fnames:
        return {}

    global_n_jets = np.concatenate([all_n_jets[f] for f in fnames], axis=0)
    global_event_weights = {}

    # Identify keys to process (variation weights)
    weight_keys = [k for k in all_raw_weights[fnames[0]].keys() if isinstance(all_raw_weights[fnames[0]][k], np.ndarray) and any(x in k for x in ['nom', 'up', 'down', 'vars'])]

    for key in weight_keys:
        raw_jet_w = np.concatenate([all_raw_weights[f][key] for f in fnames], axis=0)
        # 1. Unflatten to Event Weights
        evt_w = unflatten_event_weights(raw_jet_w, global_n_jets)

        # 2. Normalize Event Weights to mean 1.0 (ONLY for events that have jets!)
        mask = global_n_jets > 0
        evt_w_norm = np.copy(evt_w)
        if np.sum(mask) > 0:
            mean_w = np.mean(evt_w[mask], axis=0, keepdims=True) if evt_w.ndim == 2 else np.mean(evt_w[mask])
            norm_factor = 1.0 / mean_w
            evt_w_norm[mask] = evt_w[mask] * norm_factor

        global_event_weights[key] = evt_w_norm

    # Split back into per-file dictionaries
    all_event_weights = {}
    current_evt_idx = 0
    for f in fnames:
        num_evts = len(all_n_jets[f])
        all_event_weights[f] = {}
        for key in weight_keys:
            all_event_weights[f][key] = global_event_weights[key][current_evt_idx : current_evt_idx + num_evts]
        current_evt_idx += num_evts

    return all_event_weights

# ==============================================================================
# 8. Scale Factor and Uncertainty (Pass 4)
# ==============================================================================

def calculate_sf_and_unc(all_raw_weights: Dict[str, Dict[str, Any]], all_jets: Dict[str, np.ndarray], all_gen_weights: Dict[str, np.ndarray], selection_func: Any) -> Dict[str, Any]:
    """
    Computes global efficiency, scale factors, and uncertainties using RAW JET-LEVEL weights
    multiplied by generator-level event weights.
    """
    print("\n[Pass 4] Calculating global Jet Scale Factor and uncertainties...")
    fnames = list(all_raw_weights.keys())
    if not fnames:
        return {}

    global_jets = np.concatenate([all_jets[f] for f in fnames], axis=0)
    nom_mc_weights = np.concatenate([all_gen_weights[f] for f in fnames], axis=0)

    # Concatenate raw LP weights for global calculation
    global_LP_weights = {}
    for key in all_raw_weights[fnames[0]].keys():
        if isinstance(all_raw_weights[fnames[0]][key], np.ndarray):
            global_LP_weights[key] = np.concatenate([all_raw_weights[f][key] for f in fnames], axis=0)
        elif isinstance(all_raw_weights[fnames[0]][key], list):
            global_LP_weights[key] = []
            for f in fnames:
                global_LP_weights[key].extend(all_raw_weights[f][key])

    # The LJP weights in global_LP_weights are raw (not normalized).
    # We multiply them by generator weights to get the total MC weighted yield.
    for key in global_LP_weights.keys():
        if "nom" in key or "up" in key or "down" in key or "vars" in key:
            if isinstance(global_LP_weights[key], np.ndarray):
                if global_LP_weights[key].ndim == 2:
                    global_LP_weights[key] *= nom_mc_weights[:, np.newaxis]
                else:
                    global_LP_weights[key] *= nom_mc_weights

    print(f"Average Bad Match Fraction: {np.mean(global_LP_weights.get('bad_match', [0])):.3f}")

    score_cut = selection_func(global_jets)

    eff_nom = np.average(score_cut, weights=nom_mc_weights)
    eff_rw = np.average(score_cut, weights=global_LP_weights["nom"])
    sf_nominal = eff_rw / eff_nom if eff_nom > 0 else 0.0

    print(f"Nominal Eff: {eff_nom:.3f} | Corrected Eff: {eff_rw:.3f} | SF: {sf_nominal:.3f}")

    nToys = global_LP_weights["stat_vars"].shape[1]
    eff_toys = [np.average(score_cut, weights=global_LP_weights["stat_vars"][:, i]) for i in range(nToys)]
    pt_eff_toys = [np.average(score_cut, weights=global_LP_weights["pt_vars"][:, i]) for i in range(nToys)]

    eff_stat_unc = abs(np.mean(eff_toys) - eff_rw) + np.std(eff_toys)
    eff_pt_unc = abs(np.mean(pt_eff_toys) - eff_rw) + np.std(pt_eff_toys)

    sys_keys = ["sys", "bquark", "prongs", "unclust", "distortion"]
    sys_uncs = {}

    for sys_key in sys_keys:
        eff_up = np.average(score_cut, weights=global_LP_weights.get(f"{sys_key}_up", global_LP_weights["nom"]))
        eff_down = np.average(score_cut, weights=global_LP_weights.get(f"{sys_key}_down", global_LP_weights["nom"]))
        sys_uncs[sys_key] = (eff_up - eff_rw, eff_down - eff_rw)

    tot_unc_up_sq = eff_stat_unc**2 + eff_pt_unc**2
    tot_unc_down_sq = eff_stat_unc**2 + eff_pt_unc**2

    for (unc_up, unc_down) in sys_uncs.values():
        tot_unc_up_sq += max(unc_up, unc_down)**2
        tot_unc_down_sq += min(unc_up, unc_down)**2

    tot_unc_up = math.sqrt(tot_unc_up_sq)
    tot_unc_down = math.sqrt(tot_unc_down_sq)

    results = {
        "eff_nom": eff_nom,
        "eff_rw": eff_rw,
        "sf_nominal": sf_nominal,
        "unc_up": tot_unc_up,
        "unc_down": tot_unc_down,
        "sf_unc_up": tot_unc_up / eff_nom if eff_nom > 0 else 0.0,
        "sf_unc_down": tot_unc_down / eff_nom if eff_nom > 0 else 0.0,
    }

    print("\n================ FINAL RESULTS ================")
    print(f"Original Efficiency : {eff_nom:.4f}")
    print(f"Corrected Efficiency: {eff_rw:.4f}  +{tot_unc_up:.4f} / -{tot_unc_down:.4f}")
    print(f"Derived Scale Factor: {sf_nominal:.4f}  +{results['sf_unc_up']:.4f} / -{results['sf_unc_down']:.4f}")
    print("===============================================\n")

    return results

# ==============================================================================
# 9. Main Flow
# ==============================================================================

def parse_arguments():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Lund Jet Plane Reweighting on Ntuple/NanoAOD files.")
    parser.add_argument("--inputs", nargs="+", default=["/t3home/fameng/work/BosonRes/CMSSW_14_1_9/src/LundReweighting/013c4b44-92a8-42a8-ac27-c127158c4726.root"],
                        help="List of paths to the signal ROOT files.")
    parser.add_argument("--ratio", type=str, default="data/ratio_2018.root",
                        help="Path to the Lund Plane ratio correction ROOT file.")
    parser.add_argument("--fatjet", type=str, default="FatJet", help="Branch name for FatJets.")
    parser.add_argument("--pfcand", type=str, default="PFCand", help="Branch name for PF Candidates.")
    parser.add_argument("--genpart", type=str, default="GenPart", help="Branch name for Generator-level particles.")
    parser.add_argument("--fatjet_pfcand", type=str, default="FatJetPFCand", help="Branch name mapping FatJets to PFCands.")
    parser.add_argument("--genweight", type=str, default="genWeight", help="Branch name for generator-level event weight.")
    parser.add_argument("--max_process_jets", type=int, default=1000, help="Maximum jets to extract per file. Use a negative value to process all.")
    parser.add_argument("--min_pt", type=float, default=400.0, help="Minimum transverse momentum cut for jets.")
    parser.add_argument("--features", type=str, nargs="+", default=["tau1", "tau2"],
                        help="List of additional jet features to extract (e.g. tau1 tau2).")
    parser.add_argument("--selection_func", type=str, default="tau21_selections",
                        help="Name of the selection function in scripts/selections.py.")
    parser.add_argument("--triggers", type=str, nargs="+", default=["HLT_PFJet500"],
                        help="List of trigger branch names.")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Number of jets to process simultaneously to save memory.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers for processing.")
    parser.add_argument("--topology", type=str, choices=["b2b", "boost"], default="b2b",
                        help="Jet topology: 'b2b' (ttbar-like, no cross-matching) or 'boost' (H->4q like, pool matching).")
    parser.add_argument("--w_max", type=float, default=10.0, help="Maximum weight allowed for clipping to remove outliers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for toy variations.")

    args = parser.parse_args()

    # Adjust workers to not exceed the number of input files
    args.workers = min(args.workers, len(args.inputs))

    return args

def main(args):
    """
    Main CLI entry point to orchestrate the refactored LJP pipeline.
    """

    f_ratio = ROOT.TFile.Open(args.ratio)
    if not f_ratio or f_ratio.IsZombie():
        print(f"Error: Unable to load ratio file {args.ratio}")
        sys.exit(1)

    print("\nInitializing LundReweighter...")
    LP_rw = LundReweighter(f_ratio=f_ratio, use_CA=False, pf_pt_min=1.0)

    if args.seed is not None:
        np.random.seed(args.seed)

    nToys = 100
    rand_noise = np.random.normal(size=(nToys, LP_rw.h_ratio.GetNbinsX(), LP_rw.h_ratio.GetNbinsY(), LP_rw.h_ratio.GetNbinsZ()))
    pt_rand_noise = np.random.normal(size=(nToys, LP_rw.h_ratio.GetNbinsY(), LP_rw.h_ratio.GetNbinsZ(), 3))

    triggers = args.triggers

    # Pass 1: Global Distortion
    h_distortion_ratio = get_global_distortion(args.inputs, LP_rw, args, triggers)

    # Extract distortion bin contents to pass to workers
    dist_ncells = h_distortion_ratio.GetNcells()
    dist_contents = np.zeros(dist_ncells, dtype=np.float64)
    for i in range(dist_ncells):
        dist_contents[i] = h_distortion_ratio.GetBinContent(i)

    all_raw_weights = {}
    all_jets = {}
    all_n_jets = {}
    all_gen_weights = {}

    # Pass 2: Raw Weights (Multiprocessing)
    print(f"\n[Pass 2] Computing raw Lund Plane weights (using {args.workers} workers)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_pass2, fpath, args.ratio, args, triggers, rand_noise, pt_rand_noise, dist_contents): fpath for fpath in args.inputs}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            fpath = futures[future]
            fpath_res, raw_weights, jets, n_jets_per_evt, gen_w = future.result()
            print(f"  -> Pass 2 finished for {fpath_res} ({i+1}/{len(args.inputs)})")
            if len(jets) > 0 and raw_weights:
                all_raw_weights[fpath_res] = raw_weights
                all_jets[fpath_res] = jets
                all_n_jets[fpath_res] = n_jets_per_evt
                all_gen_weights[fpath_res] = gen_w

    if not all_raw_weights:
        print("\nPipeline aborted: No matching jets passed in any input file.")
        sys.exit(0)

    # Pass 3: Normalized Event Weights for MC Calibration
    # These are the weights to use when filling histograms of event-level observables.
    all_event_weights = get_normalized_event_weights(all_raw_weights, all_n_jets)

    # Pass 4: Final SF and Uncertainties
    # These are calculated using raw jet-level weights multiplied by gen weights.
    # Load the selection function from selections.py
    try:
        selections_mod = importlib.import_module("selections")
        selection_func = getattr(selections_mod, args.selection_func)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not load selection function '{args.selection_func}' from scripts/selections.py: {e}")
        sys.exit(1)
    _ = calculate_sf_and_unc(all_raw_weights, all_jets, all_gen_weights, selection_func)

    f_ratio.Close()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
