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

import numpy as np
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

# ==============================================================================
# 1. Matching Logic
# ==============================================================================

def match_jet_to_gen(jet: Any, f1: Any, f2: Any, b_quark: Any, dr_cut: float = 0.8) -> bool:
    """
    Check if a jet matches at least two of the three gen particles (f1, f2, b_quark).
    Equivalent to check_matching in example_NanoV15.py.
    """
    count = 0
    if f1 and deltaR(jet.eta, jet.phi, f1.eta, f1.phi) < dr_cut:
        count += 1
    if f2 and deltaR(jet.eta, jet.phi, f2.eta, f2.phi) < dr_cut:
        count += 1
    if b_quark and deltaR(jet.eta, jet.phi, b_quark.eta, b_quark.phi) < dr_cut:
        count += 1
    return count > 1

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

    try:
        GenPartsColl = Collection(event, genpart_branch)
    except RuntimeError:
        if verbose:
            print(f"Warning: Branch '{genpart_branch}' not found in event.")
        return (None,) * 10

    top = anti_top = W = anti_W = None
    q1a = q1b = b1 = q2a = q2b = b2 = None

    # First pass: Find top and W bosons
    for genPart in GenPartsColl:
        if abs(genPart.pdgId) == top_ID and isFinal(genPart):
            if genPart.pdgId > 0:
                top = genPart if top is None else top
            else:
                anti_top = genPart if anti_top is None else anti_top

        if abs(genPart.pdgId) == W_ID and isFinal(genPart):
            if genPart.pdgId > 0:
                W = genPart if W is None else W
            else:
                anti_W = genPart if anti_W is None else anti_W

    # Second pass: Find quarks from W and top decays
    for genPart in GenPartsColl:
        m = genPart.genPartIdxMother
        mother = GenPartsColl[m] if m >= 0 else None

        # Quarks/leptons from W decay
        if abs(genPart.pdgId) <= MAXLEP_ID and m >= 0:
            if mother is W:
                if genPart.pdgId > 0:
                    q1a = genPart if q1a is None else q1a
                else:
                    q1b = genPart if q1b is None else q1b
            elif mother is anti_W:
                if genPart.pdgId > 0:
                    q2a = genPart if q2a is None else q2a
                else:
                    q2b = genPart if q2b is None else q2b

        # b quarks from top decay
        if abs(genPart.pdgId) == B_ID and m >= 0:
            if mother is top:
                b1 = genPart if b1 is None else b1
            elif mother is anti_top:
                b2 = genPart if b2 is None else b2

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
        try:
            # Using logical OR across triggers
            pass_trigger = pass_trigger or inTree.readBranch(trig)
        except RuntimeError:
            # Branch might not exist
            pass

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
    max_events: int = 5000,
    jet_min_pt: float = 400.0,
    triggers: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[List[List[float]]]]:
    """
    Processes the input ROOT tree, applying selections and extracting arrays
    for jets, gen quarks, and PF candidates.

    Args:
        input_file (ROOT.TFile): Opened ROOT file.
        fatjet_branch (str): Branch name for FatJets.
        pfcand_branch (str): Branch name for PFCands.
        genpart_branch (str): Branch name for GenParts.
        fatjet_pfcand_branch (str): Branch name linking FatJets and PFCands.
        max_events (int): Max events to process.
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
    print(f"Total entries available: {TTree.GetEntries()}")

    nEvents = 0
    jets_list = []
    quarks_list = []
    cands_list = []

    for entry in range(inTree.entries):
        if entry % 10000 == 0:
            print(f"--- Processing Event {entry} | Saved {nEvents} jets so far")

        event = Event(inTree, entry)

        if not apply_selections(inTree, triggers):
            continue

        try:
            AK8Jets = Collection(event, fatjet_branch)
            PFCands = Collection(event, pfcand_branch)
            FatJetPFCands = Collection(event, fatjet_pfcand_branch)
        except RuntimeError:
            continue

        if len(AK8Jets) == 0 or AK8Jets[0].pt < jet_min_pt:
            continue

        # Extract gen level topology
        gen_parts = extract_gen_particles(event, genpart_branch=genpart_branch)
        _, _, _, _, q1a, q1b, b1, q2a, q2b, b2 = gen_parts

        my_jet = None
        for i, jet in enumerate(AK8Jets):
            jet.idx = i
            if jet.pt > jet_min_pt and abs(jet.eta) < 2.4:
                # Check truth-matching to ensure it's a genuine multi-prong jet
                if match_jet_to_gen(jet, q1a, q1b, b1) or match_jet_to_gen(jet, q2a, q2b, b2):
                    my_jet = jet
                    break

        if my_jet is None:
            continue

        nEvents += 1

        # Extract Jet features
        eps = 1e-6
        tau21 = getattr(my_jet, "tau2", 0) / (getattr(my_jet, "tau1", 0) + eps)
        pnet_xqq = getattr(my_jet, "particleNetLegacy_Xqq", -1.0)

        jets_list.append([
            my_jet.pt, my_jet.eta, my_jet.phi, getattr(my_jet, "msoftdrop", 0), tau21, pnet_xqq
        ])

        # Extract Gen Quark features
        def parse_part(p: Any) -> List[float]:
            return [p.pt, p.eta, p.phi, float(p.pdgId)] if p else [0.0, 0.0, 0.0, 0.0]

        quark_group = [
            parse_part(q1a), parse_part(q1b), parse_part(b1),
            parse_part(q2a), parse_part(q2b), parse_part(b2)
        ]
        quarks_list.append(quark_group)

        # Extract PF Candidates for the selected jet
        jet_PFCands = []
        for cand_map in FatJetPFCands:
            if cand_map.jetIdx == my_jet.idx:
                idx = cand_map.pfCandIdx
                cand_obj = PFCands[idx]
                cand_vec = ROOT.Math.PtEtaPhiMVector(
                    cand_obj.pt, cand_obj.eta, cand_obj.phi, getattr(cand_obj, "mass", 0)
                )
                jet_PFCands.append([cand_vec.Px(), cand_vec.Py(), cand_vec.Pz(), cand_vec.E()])

        cands_list.append(jet_PFCands)

        if nEvents >= max_events:
            break

    return np.array(jets_list), np.array(quarks_list), cands_list

# ==============================================================================
# 5. Weight and Scale Factor Calculation
# ==============================================================================

def calculate_weights_and_sf(
    jets: np.ndarray,
    quarks: np.ndarray,
    cands: List[List[List[float]]],
    ratio_file_path: str,
    tau21_cut: float = 0.4,
    chunk_size: int = 5000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes the Lund Plane weights, determines the efficiency of a substructure cut,
    and estimates Scale Factors (SF) with full uncertainty breakdown.

    Args:
        jets (np.ndarray): Extracted jets array.
        quarks (np.ndarray): Extracted gen quarks array.
        cands (List[List[List[float]]]): PF candidates.
        ratio_file_path (str): Path to the correction root file.
        tau21_cut (float): Substructure cut threshold for evaluation.

    Returns:
        Dict[str, Any]: Mapping of computed metrics and uncertainties.
    """
    f_ratio = ROOT.TFile.Open(ratio_file_path)
    if not f_ratio or f_ratio.IsZombie():
        raise RuntimeError(f"Could not open Lund ratio file at {ratio_file_path}")

    print("\\nInitializing LundReweighter...")
    LP_rw = LundReweighter(f_ratio=f_ratio)

    ak8_jets = jets[:, :4]
    gen_parts_eta_phi = quarks[:, :, 1:3]
    gen_parts_pdg_id = quarks[:, :, 3]

    # Assume unit weights for MC events in this example
    nom_weights = np.ones(len(cands))

    print("Computing Lund Plane weights (this may take a moment)...")

    if seed is not None:
        np.random.seed(seed)

    nToys = 100
    rand_noise = np.random.normal(size=(nToys, LP_rw.h_ratio.GetNbinsX(), LP_rw.h_ratio.GetNbinsY(), LP_rw.h_ratio.GetNbinsZ()))
    pt_rand_noise = np.random.normal(size=(nToys, LP_rw.h_ratio.GetNbinsY(), LP_rw.h_ratio.GetNbinsZ(), 3))

    num_jets = len(jets)
    num_chunks = math.ceil(num_jets / chunk_size)

    # Pass 1: Global LP distribution for distortion systematic (requires reclustering)
    print("Pass 1/2: Computing global Lund Plane distribution for distortion systematic...")
    h_lp_signal = LP_rw.h_mc.Clone("h_lp_signal")
    h_lp_signal.Reset()
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_jets)
        print(f"  Analysing chunk {i+1}/{num_chunks} for distortion...")
        for j in range(start, end):
            # get_splittings_and_matching performs the expensive reclustering
            reclust_nom, _, _ = LP_rw.get_splittings_and_matching(cands[j], gen_parts_eta_phi[j], ak8_jets[j])
            if not reclust_nom.badmatch:
                LP_rw.fill_lund_plane(h_lp_signal, reclust_obj=reclust_nom)
    
    h_dummy = LP_rw.h_mc.Clone("h_dummy")
    h_dummy.Reset()
    h_distortion_ratio = LP_rw.make_LP_ratio(LP_rw.h_mc, h_dummy, h_lp_signal)
    # cleanup_ratio is available from utils.Utils (via LundReweighter import *)
    try:
        from utils.LundReweighter import cleanup_ratio
        cleanup_ratio(h_distortion_ratio, h_min=0.2, h_max=5.0)
    except ImportError:
        pass # If not available, skip cleanup as it's a safety measure

    # Pass 2: Calculate weights with global distortion ratio
    print("\nPass 2/2: Computing Lund Plane weights...")
    LP_weights_combined = {}

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_jets)
        print(f"  Processing chunk {i+1}/{num_chunks}...")

        chunk_cands = cands[start:end]
        chunk_eta_phi = gen_parts_eta_phi[start:end]
        chunk_jets = ak8_jets[start:end]
        chunk_pdg_id = gen_parts_pdg_id[start:end]

        # Call with distortion_sys=False to avoid local bias, we apply it manually below
        chunk_weights = LP_rw.get_all_weights(
            chunk_cands, chunk_eta_phi, chunk_jets, gen_parts_pdg_ids=chunk_pdg_id,
            nToys=nToys, rand_noise=rand_noise, pt_rand_noise=pt_rand_noise, 
            normalize=False, distortion_sys=False
        )

        # Apply global distortion systematic manually for this chunk
        dist_up = np.zeros(len(chunk_jets))
        dist_down = np.zeros(len(chunk_jets))
        for j in range(len(chunk_jets)):
            # Reclustering again is unfortunate but necessary for exact matching
            reclust_nom, _, _ = LP_rw.get_splittings_and_matching(chunk_cands[j], chunk_eta_phi[j], chunk_jets[j])
            
            # Use LP_rw.reweight_lund_plane to get the distortion weight
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

    # Concatenate the accumulated numpy arrays
    for key, val in LP_weights_combined.items():
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], np.ndarray):
            LP_weights_combined[key] = np.concatenate(val, axis=0)

    # Manual Normalization
    print("Normalizing combined weights...")
    for key in LP_weights_combined.keys():
        if 'nom' in key or 'up' in key or 'down' in key or 'vars' in key:
            if isinstance(LP_weights_combined[key], np.ndarray):
                LP_weights_combined[key] = LP_rw.normalize_weights(
                    LP_weights_combined[key],
                    n_prongs=LP_weights_combined['n_prongs'],
                    pt_norm=True,
                    ak8_pts=ak8_jets[:, 0]
                )

    LP_weights = LP_weights_combined

    # Multiply LP weights with event weights
    for key in LP_weights.keys():
        if "nom" in key or "up" in key or "down" in key or "vars" in key:
            if isinstance(LP_weights[key], np.ndarray):
                if LP_weights[key].ndim == 2:
                    LP_weights[key] *= nom_weights[:, np.newaxis]
                else:
                    LP_weights[key] *= nom_weights

    print(f"Average Bad Match Fraction: {np.mean(LP_weights.get('bad_match', [0])):.3f}")

    # ===============================
    # Efficiency & SF calculation
    # ===============================
    tau21 = jets[:, 4]
    score_cut = tau21 < tau21_cut

    eff_nom = np.average(score_cut, weights=nom_weights)
    eff_rw = np.average(score_cut, weights=LP_weights["nom"])
    sf_nominal = eff_rw / eff_nom if eff_nom > 0 else 0.0

    print(f"Nominal Eff: {eff_nom:.3f} | Corrected Eff: {eff_rw:.3f} | SF: {sf_nominal:.3f}")

    # ===============================
    # Uncertainty Calculation
    # ===============================

    # 1. Statistical and Pt extrapolation uncertainties (via Toys)
    nToys = LP_weights["stat_vars"].shape[1]
    eff_toys = [np.average(score_cut, weights=LP_weights["stat_vars"][:, i]) for i in range(nToys)]
    pt_eff_toys = [np.average(score_cut, weights=LP_weights["pt_vars"][:, i]) for i in range(nToys)]

    eff_stat_unc = abs(np.mean(eff_toys) - eff_rw) + np.std(eff_toys)
    eff_pt_unc = abs(np.mean(pt_eff_toys) - eff_rw) + np.std(pt_eff_toys)

    # 2. Up/Down Systematic Uncertainties
    sys_keys = ["sys", "bquark", "prongs", "unclust", "distortion"]
    sys_uncs = {}

    for sys_key in sys_keys:
        eff_up = np.average(score_cut, weights=LP_weights.get(f"{sys_key}_up", LP_weights["nom"]))
        eff_down = np.average(score_cut, weights=LP_weights.get(f"{sys_key}_down", LP_weights["nom"]))
        sys_uncs[sys_key] = (eff_up - eff_rw, eff_down - eff_rw)

    # 3. Summing in Quadrature
    tot_unc_up_sq = eff_stat_unc**2 + eff_pt_unc**2
    tot_unc_down_sq = eff_stat_unc**2 + eff_pt_unc**2

    for (unc_up, unc_down) in sys_uncs.values():
        tot_unc_up_sq += max(unc_up, unc_down)**2
        tot_unc_down_sq += min(unc_up, unc_down)**2

    tot_unc_up = math.sqrt(tot_unc_up_sq)
    tot_unc_down = math.sqrt(tot_unc_down_sq)

    f_ratio.Close()

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
# 6. Main Flow
# ==============================================================================

def main():
    """
    Main CLI entry point to orchestrate the refactored LJP pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run Lund Jet Plane Reweighting on an Ntuple/NanoAOD file."
    )
    parser.add_argument("--input", type=str, default="/t3home/fameng/work/BosonRes/CMSSW_14_1_9/src/LundReweighting/013c4b44-92a8-42a8-ac27-c127158c4726.root",
                        help="Path to the signal ROOT file.")
    parser.add_argument("--ratio", type=str, default="data/ratio_2018.root",
                        help="Path to the Lund Plane ratio correction ROOT file.")
    parser.add_argument("--fatjet", type=str, default="FatJet",
                        help="Branch name for large radius jets (FatJets).")
    parser.add_argument("--pfcand", type=str, default="PFCand",
                        help="Branch name for PF Candidates.")
    parser.add_argument("--genpart", type=str, default="GenPart",
                        help="Branch name for Generator-level particles.")
    parser.add_argument("--fatjet_pfcand", type=str, default="FatJetPFCand",
                        help="Branch name mapping FatJets to PFCands.")
    parser.add_argument("--max_events", type=int, default=1000,
                        help="Maximum events to evaluate.")
    parser.add_argument("--min_pt", type=float, default=400.0,
                        help="Minimum transverse momentum cut for jets.")
    parser.add_argument("--tau21_cut", type=float, default=0.4,
                        help="Tau21 substructure cut to evaluate for the SF.")
    parser.add_argument("--chunk_size", type=int, default=5000,
                        help="Number of jets to process simultaneously to save memory.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for toy variations.")

    args = parser.parse_args()

    print(f"Opening input file: {args.input}")
    f_sig = ROOT.TFile.Open(args.input)
    if not f_sig or f_sig.IsZombie():
        print(f"Error: Unable to load input file {args.input}")
        sys.exit(1)

    triggers = [
        "HLT_PFHT890",
        "HLT_PFHT1050",
        "HLT_PFJet450",
        "HLT_PFJet500",
    ]

    print("\n[1/2] Processing inputs and extracting arrays...")
    jets, quarks, cands = process_inputs(
        input_file=f_sig,
        fatjet_branch=args.fatjet,
        pfcand_branch=args.pfcand,
        genpart_branch=args.genpart,
        fatjet_pfcand_branch=args.fatjet_pfcand,
        max_events=args.max_events,
        jet_min_pt=args.min_pt,
        triggers=triggers
    )

    f_sig.Close()

    if len(jets) == 0:
        print("Pipeline aborted: No matching jets passed the predefined criteria.")
        sys.exit(0)

    print(f"Successfully extracted {len(jets)} jets.")

    print("\n[2/2] Calculating Lund Weights and Scale Factors...")
    _ = calculate_weights_and_sf(
        jets=jets,
        quarks=quarks,
        cands=cands,
        ratio_file_path=args.ratio,
        tau21_cut=args.tau21_cut,
        chunk_size=args.chunk_size,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
