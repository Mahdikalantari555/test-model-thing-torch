"""Deterministic benchmark workloads and offline fixtures for the RecaMem tournament.

Covers:
1. Mechanistic synthetic tasks: `MQAR` (`4, 16, 32, 64` KV pairs) and exact delayed string copy.
2. Language modeling (`train`, `val_cal`, `test_eval`) + length extrapolation (`256 -> 512..4096`).
3. Contrastive syntax probes (`evals/grammar_probes.jsonl` + BLiMP-style minimal pairs).
4. Dynamic/shuffled/unseen-label System-1 intent decisions + out-of-scope (`null`) rejection.
5. 5-way cost-weighted action/tool routing with valid and malformed/unpermitted tool calls.
6. 5-domain sequential continual learning + entity-slot contradiction updates.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.model.benchmarks import find_contamination
from src.model.cores.calibration import assert_disjoint_splits
from src.model.tournament.large_workloads import (
    build_large_blimp_probes,
    build_large_continual_suite,
    build_large_decision_suite,
    build_large_routing_suite,
    compute_corpus_unigram_bpb,
    ensure_tinystories_real_data_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE_DIR = REPO_ROOT / "evals" / "tournament_fixtures"
GRAMMAR_PROBES_PATH = REPO_ROOT / "evals" / "grammar_probes.jsonl"


@dataclass
class MQARExample:
    example_id: str
    num_pairs: int
    prefix_tokens: List[int]
    queries: List[Tuple[int, int]]  # list of (key_token_id, expected_val_token_id)


@dataclass
class CopyExample:
    example_id: str
    source_tokens: List[int]
    delay_tokens: List[int]
    delim_token: int


@dataclass
class WorkloadBundle:
    status: str  # "ready" | "blocked"
    dataset_source: str  # "real" | "fixture" | "missing_real"
    reason: Optional[str] = None
    corpus_metadata: Dict[str, Any] = field(default_factory=dict)
    item_counts: Dict[str, Any] = field(default_factory=dict)
    lm_train: Dict[str, str] = field(default_factory=dict)
    lm_val_cal: Dict[str, str] = field(default_factory=dict)
    lm_test_eval: Dict[str, str] = field(default_factory=dict)
    extrapolation_docs: Dict[int, List[str]] = field(default_factory=dict)
    grammar_probes: List[Dict[str, Any]] = field(default_factory=list)
    mqar_splits: Dict[int, Dict[str, List[MQARExample]]] = field(default_factory=dict)
    copy_splits: Dict[str, List[CopyExample]] = field(default_factory=dict)
    decisions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    routing: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    continual_domains: List[Dict[str, Any]] = field(default_factory=list)
    contradiction_pairs: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Mechanistic Synthetic Generators (MQAR & Delayed Copy)
# --------------------------------------------------------------------------- #
def generate_mqar_examples(
    num_examples: int,
    num_kv_pairs: int = 16,
    vocab_low: int = 16,
    vocab_high: int = 120,
    distractor_gap: int = 0,
    num_queries: int = 4,
    allow_overwrite: bool = True,
    seed: int = 1337,
) -> List[MQARExample]:
    """Generate Multi-Query Associative Recall (`MQAR`) sequences.

    Uses a shared 24-entity key vocabulary (`16..39`) and 20-value vocabulary (`80..99`)
    across `num_kv_pairs in {4, 16, 32, 64}` so higher pair counts test both
    associative capacity and repeated key-overwrite replacement (`latest_map[k] = v`)
    without introducing unseen key token IDs at evaluation time.
    """
    rng = random.Random(int(seed) + 997 * int(num_kv_pairs))
    key_pool = list(range(vocab_low, vocab_low + 24))
    val_start = max(vocab_low + 24, min(80, vocab_high - 28))
    val_pool = list(range(val_start, val_start + 20))
    distractor_pool = list(range(val_start + 20, val_start + 28))

    examples: List[MQARExample] = []
    n_req = int(num_kv_pairs)
    for idx in range(num_examples):
        if n_req <= len(key_pool):
            keys = rng.sample(key_pool, n_req)
        else:
            keys = rng.sample(key_pool, len(key_pool)) + [
                rng.choice(key_pool) for _ in range(n_req - len(key_pool))
            ]
            rng.shuffle(keys)
        vals = [rng.choice(val_pool) for _ in range(n_req)]

        pairs: List[Tuple[int, int]] = list(zip(keys, vals))
        if allow_overwrite and 4 <= n_req <= len(key_pool):
            upd_key = keys[0]
            alt_vals = [v for v in val_pool if v != vals[0]]
            upd_val = rng.choice(alt_vals)
            insert_pos = rng.randint(max(1, n_req // 2), n_req)
            pairs.insert(insert_pos, (upd_key, upd_val))

        seq: List[int] = []
        latest_map: Dict[int, int] = {}
        for k, v in pairs:
            seq.extend([k, v])
            latest_map[k] = v
            for _ in range(distractor_gap):
                seq.append(rng.choice(distractor_pool))

        uniq_keys = list(latest_map.keys())
        q_count = min(int(num_queries), len(uniq_keys))
        if allow_overwrite and 4 <= n_req <= len(key_pool):
            chosen_keys = [keys[0]] + rng.sample(
                [k for k in uniq_keys if k != keys[0]], max(0, q_count - 1)
            )
            rng.shuffle(chosen_keys)
        else:
            chosen_keys = rng.sample(uniq_keys, q_count)
        queries = [(qk, latest_map[qk]) for qk in chosen_keys]
        examples.append(
            MQARExample(
                example_id=f"mqar_{num_kv_pairs}_{idx:04d}",
                num_pairs=n_req,
                prefix_tokens=seq,
                queries=queries,
            )
        )
    return examples


def mqar_example_to_lm_sequence(example: MQARExample, sep_token: int = 3, eos_token: int = 1) -> Tuple[List[int], List[int]]:
    """Convert an `MQARExample` into `(token_ids, answer_positions)` for causal training/eval.

    Sequence format:
      `prefix_tokens + [sep_token, qk_1, qv_1, sep_token, qk_2, qv_2, ..., eos_token]`
    `answer_positions` holds the sequence index `p` where `token_ids[p] == qv_j`
    (predicted by `logits[p - 1]`).
    """
    seq = list(example.prefix_tokens)
    ans_positions: List[int] = []
    for qk, qv in example.queries:
        seq.append(int(sep_token))
        seq.append(int(qk))
        ans_positions.append(len(seq))
        seq.append(int(qv))
    seq.append(int(eos_token))
    return seq, ans_positions


def generate_exact_copy_examples(
    num_examples: int,
    span_len: int = 8,
    delay_len: int = 16,
    vocab_low: int = 20,
    vocab_high: int = 52,
    delim_token: int = 121,
    filler_token: int = 120,
    seed: int = 2026,
) -> List[CopyExample]:
    rng = random.Random(int(seed))
    examples: List[CopyExample] = []
    for idx in range(num_examples):
        src = [rng.randint(vocab_low, vocab_high - 1) for _ in range(span_len)]
        delay = [filler_token] * int(delay_len)
        examples.append(
            CopyExample(
                example_id=f"copy_{span_len}_{delay_len}_{idx:04d}",
                source_tokens=src,
                delay_tokens=delay,
                delim_token=delim_token,
            )
        )
    return examples


def copy_example_to_lm_sequence(example: CopyExample, eos_token: int = 1) -> Tuple[List[int], List[int]]:
    seq = list(example.source_tokens) + list(example.delay_tokens) + [example.delim_token]
    ans_positions: List[int] = []
    for tok in example.source_tokens:
        ans_positions.append(len(seq))
        seq.append(int(tok))
    seq.append(int(eos_token))
    return seq, ans_positions


# --------------------------------------------------------------------------- #
# Bundled Offline Fixture Materializer (Zero Split Contamination)
# --------------------------------------------------------------------------- #
def _build_disjoint_prose_corpora() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Construct strictly disjoint technical corpora (`train`, `val_cal`, `test_eval`).

    Every split uses distinct subject vocabularies and sentence templates so 64-byte
    sliding-window contamination checks (`find_contamination`) report zero overlap.
    """
    train_topics = [
        ("Synthetic Aperture Radar", "microwave backscatter", "interferometric phase coherence", "surface deformation"),
        ("Multispectral Land Imaging", "near-infrared reflectance", "chlorophyll red-edge slope", "canopy vigor"),
        ("Thermal Infrared Radiometry", "emissivity calibration", "brightness temperature", "evapotranspiration flux"),
        ("Hyperspectral Unmixing", "endmember abundance", "convex simplex projection", "mineral mapping"),
        ("Lidar Point Cloud Altimetry", "waveform return timing", "digital terrain filtering", "biomass estimation"),
        ("Ocean Color Spectrophotometry", "phytoplankton pigment", "atmospheric Rayleigh correction", "coastal turbidity"),
        ("Gravimetric Orbit Perturbation", "geopotential harmonics", "inter-satellite ranging", "aquifer mass balance"),
        ("Atmospheric Limb Sounding", "stratospheric ozone profile", "rotational absorption lines", "radiative forcing"),
    ]
    train_docs: Dict[str, str] = {}
    for i, (topic, m1, m2, app) in enumerate(train_topics):
        paragraphs = [
            f"Document TRN-{i:02d} analyzes {topic} pipelines for orbital earth observation.",
            f"Calibrated sensors quantify {m1} across calibrated swaths before computing {m2}.",
            f"Downstream assimilation modules translate those measurements into {app} forecasts.",
            f"State-space estimators update fast weights at step {100 + i * 7} while bounding numerical drift.",
        ]
        train_docs[f"train_doc_{i:02d}"] = " ".join(paragraphs * 4)

    val_topics = [
        ("Distributed Consensus Clocks", "monotonic lamport timestamps", "quorum lease renewal"),
        ("Acoustic Doppler Velocimetry", "backscattered ultrasonic pulses", "estuarine shear stress"),
        ("Solid-State Battery Kinetics", "interfacial lithium diffusivity", "dendrite suppression"),
        ("Cryogenic Calorimeter Readout", "transition-edge bolometers", "phonon thermal conductance"),
    ]
    val_docs: Dict[str, str] = {}
    for i, (topic, k1, k2) in enumerate(val_topics):
        val_docs[f"val_doc_{i:02d}"] = (
            f"Validation record VAL-{i:02d} examines {topic} under controlled laboratory conditions. "
            f"Instruments sample {k1} continuously to verify {k2} across independent trials. "
        ) * 3

    test_topics = [
        ("Magnetotelluric Subsurface Sounding", "telluric impedance tensors", "crustal conductivity anomalies"),
        ("Autonomous Celestial Navigation", "star-tracker quaternion attitude", "ephemeris residual bounds"),
        ("Enzymatic Michaelis Menten Kinetics", "allosteric cofactor binding", "substrate turnover plateaus"),
        ("Packet-Switched Congestion Control", "explicit queue marking", "bottleneck bandwidth probing"),
    ]
    test_docs: Dict[str, str] = {}
    for i, (topic, k1, k2) in enumerate(test_topics):
        test_docs[f"test_doc_{i:02d}"] = (
            f"Evaluation dossier TST-{i:02d} characterizes {topic} across unseen operational regimes. "
            f"Analysts measure {k1} with strict error bars to isolate {k2} during stress benchmarks. "
        ) * 3

    return train_docs, val_docs, test_docs


def _build_blimp_probes() -> List[Dict[str, Any]]:
    return [
        {
            "id": "blimp-anaphor-01",
            "phenomenon": "anaphor_agreement",
            "context": "The satellite calibrated ",
            "correct": "itself before eclipse.",
            "incorrect": ["themselves before eclipse.", "herself before eclipse."],
        },
        {
            "id": "blimp-det-01",
            "phenomenon": "determiner_noun_agreement",
            "context": "Engineers inspected ",
            "correct": "these sensors carefully.",
            "incorrect": ["this sensors carefully.", "that sensors carefully."],
        },
        {
            "id": "blimp-npi-01",
            "phenomenon": "npi_licensing",
            "context": "No ground station has ",
            "correct": "ever lost telemetry.",
            "incorrect": ["often lost any telemetry ever.", "already ever lost telemetry."],
        },
        {
            "id": "blimp-long-sva-01",
            "phenomenon": "long_distance_sva",
            "context": "The calibration tables that the flight computer near the antennas ",
            "correct": "uses are verified.",
            "incorrect": ["use is verified.", "using be verified."],
        },
    ]


def _build_decision_fixtures() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    in_domain_catalog = [
        ("orbit_maneuver", "Compute thruster delta-v burn for orbital altitude adjustment"),
        ("spectral_index", "Calculate normalized vegetation reflectance ratio from red and NIR bands"),
        ("cloud_masking", "Filter opaque cumulus pixels from optical surface imagery"),
        ("telemetry_downlink", "Schedule X-band ground station contact window for packet dump"),
        ("thermal_anomaly", "Detect wildfire hotspot temperatures from infrared radiance"),
        ("sensor_calibration", "Update dark-current radiometric gain coefficients"),
    ]
    templates = {
        "orbit_maneuver": [
            "Plan a thruster burn to raise perigee altitude by four kilometers",
            "Adjust spacecraft orbit inclination with an impulsive delta-v firing",
            "Compute firing duration for station-keeping maneuver thruster",
        ],
        "spectral_index": [
            "Derive chlorophyll vegetation ratio using near-infrared and red channels",
            "Evaluate leaf canopy index from multispectral reflectance bands",
            "Compute NDVI surface greenness map across agricultural parcels",
        ],
        "cloud_masking": [
            "Remove thick cloud and shadow pixels before compositing optical scenes",
            "Flag overcast atmospheric pixels in visible imagery",
            "Mask cirrus and cumulus contamination from reflectance tiles",
        ],
        "telemetry_downlink": [
            "Book Svalbard X-band antenna pass to transmit stored recorder packets",
            "Transmit buffered housekeeping frames during ground station visibility",
            "Downlink compressed payload telemetry over high-rate radio link",
        ],
        "thermal_anomaly": [
            "Locate active volcanic and wildfire heat signatures in thermal band",
            "Identify elevated surface brightness temperature hotspots at night",
            "Trigger alert for extreme infrared radiance spike on forest perimeter",
        ],
        "sensor_calibration": [
            "Recalibrate detector dark offset and flat-field radiometric gain",
            "Measure onboard blackbody reference to correct gain drift",
            "Update focal-plane calibration lookup table before imaging",
        ],
    }

    rng = random.Random(42)
    idx_counter = 0
    for split_name, repeat_idx in (("train", 0), ("val_cal", 1), ("test_in_domain", 2)):
        for gold_key, _ in in_domain_catalog:
            query_base = templates[gold_key][repeat_idx]
            for variant in range(3):
                idx_counter += 1
                k_opts = rng.choice([3, 4, 5])
                distractors = [item for item in in_domain_catalog if item[0] != gold_key]
                chosen = [(gold_key, dict(in_domain_catalog)[gold_key])] + rng.sample(distractors, k_opts - 1)
                rng.shuffle(chosen)
                gold_idx = next(i for i, (k, _) in enumerate(chosen) if k == gold_key)
                rows.append(
                    {
                        "id": f"dec_{split_name}_{idx_counter:04d}",
                        "split": split_name,
                        "query": f"{query_base} (case {idx_counter})",
                        "option_names": [k for k, _ in chosen],
                        "option_descriptions": [desc for _, desc in chosen],
                        "gold_index": gold_idx,
                        "unseen_domain": False,
                    }
                )
        # Add explicit out-of-scope (`null`, gold_index = -1) examples in every split!
        oos_queries = [
            f"Bake sourdough bread with seventy percent hydration ({split_name} a)",
            f"Translate medieval Latin poetry into conversational Japanese ({split_name} b)",
            f"Tune acoustic guitar intonation at the twelfth fret ({split_name} c)",
        ]
        for oq in oos_queries:
            idx_counter += 1
            chosen = rng.sample(in_domain_catalog, 4)
            rows.append(
                {
                    "id": f"dec_{split_name}_oos_{idx_counter:04d}",
                    "split": split_name,
                    "query": oq,
                    "option_names": [k for k, _ in chosen],
                    "option_descriptions": [desc for _, desc in chosen],
                    "gold_index": -1,
                    "unseen_domain": False,
                }
            )

    # Shifted / unseen-label evaluation split: completely new option domains & rephrased labels!
    unseen_catalog = [
        ("maritime_ais_track", "Correlate ship transponder beacons with coastal radar echoes"),
        ("soil_moisture_sar", "Invert dielectric radar backscatter into volumetric root-zone water"),
        ("glacier_velocity", "Track crevasse speckle displacement to measure ice-sheet flow speed"),
        ("ionospheric_tec", "Estimate total electron content delay from dual-frequency GNSS phase"),
    ]
    unseen_queries = [
        ("maritime_ais_track", "Match vessel radio beacon positions against marine surveillance radar"),
        ("soil_moisture_sar", "Estimate agricultural root-zone water content from microwave dielectric return"),
        ("glacier_velocity", "Measure Antarctic ice stream flow rate by tracking surface feature offsets"),
        ("ionospheric_tec", "Compute plasma phase delay across dual-band navigation carrier signals"),
    ]
    for rep in range(3):
        for gold_key, q_text in unseen_queries:
            idx_counter += 1
            ordered = list(unseen_catalog)
            rng.shuffle(ordered)
            gold_idx = next(i for i, (k, _) in enumerate(ordered) if k == gold_key)
            rows.append(
                {
                    "id": f"dec_shifted_{idx_counter:04d}",
                    "split": "test_shifted_unseen",
                    "query": f"{q_text} [trial {rep}]",
                    "option_names": [k for k, _ in ordered],
                    "option_descriptions": [d for _, d in ordered],
                    "gold_index": gold_idx,
                    "unseen_domain": True,
                }
            )
    # Unseen-split OOS queries
    for rep in range(3):
        idx_counter += 1
        ordered = list(unseen_catalog)
        rng.shuffle(ordered)
        rows.append(
            {
                "id": f"dec_shifted_oos_{idx_counter:04d}",
                "split": "test_shifted_unseen",
                "query": f"Draft corporate tax depreciation schedules for retail inventory [trial {rep}]",
                "option_names": [k for k, _ in ordered],
                "option_descriptions": [d for _, d in ordered],
                "gold_index": -1,
                "unseen_domain": True,
            }
        )
    return rows


def _build_routing_fixtures() -> List[Dict[str, Any]]:
    schema = {
        "allowed_tools": ["calculator", "orbit_lookup"],
        "tools": {
            "calculator": {
                "required": ["expression"],
                "properties": {"expression": {"type": "string"}},
                "additionalProperties": False,
            },
            "orbit_lookup": {
                "required": ["norad_id"],
                "properties": {"norad_id": {"type": "integer"}},
                "additionalProperties": False,
            },
        },
    }
    cases = [
        ("generate", "Write a concise two-sentence summary of linear state-space recurrence.", 0.20, None),
        ("classify", "Which of the provided intent options best classifies this anomaly report?", 0.30, None),
        ("retrieve", "According to our stored memory facts, what is the Sentinel revisit period?", 0.88, None),
        (
            "tool",
            "Use the calculator tool to compute 144 * 29.",
            0.25,
            {"name": "calculator", "arguments": {"expression": "144 * 29"}},
        ),
        (
            "abstain",
            "What is the classified private launch abort code never provided in context?",
            0.12,
            None,
        ),
        # Unsafe/malformed tool call where safety guard MUST block `tool` and force `abstain`!
        (
            "abstain",
            "Execute calculator tool with invalid boolean payload.",
            0.20,
            {"name": "calculator", "arguments": {"expression": True}},
        ),
        (
            "abstain",
            "Run unpermitted shell_delete tool on telemetry archive.",
            0.20,
            {"name": "shell_delete", "arguments": {"target": "all"}},
        ),
    ]
    rows: List[Dict[str, Any]] = []
    cid = 0
    for split_name in ("train", "val_cal", "test_eval"):
        for rep in range(3):
            for gold_action, base_q, ret_sim, tool_call in cases:
                cid += 1
                rows.append(
                    {
                        "id": f"rt_{split_name}_{cid:04d}",
                        "split": split_name,
                        "query": f"{base_q} ({split_name}-{rep})",
                        "gold_action": gold_action,
                        "retrieval_similarity": ret_sim,
                        "tool_schema": schema,
                        "proposed_tool_call": tool_call,
                    }
                )
    return rows


def _build_continual_and_contradiction_fixtures() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    domains = [
        {
            "domain_id": 0,
            "domain_name": "satellites",
            "facts": [
                {"entity_key": "sentinel2 revisit", "text": "Sentinel2 revisit is 5 days", "answer": "5 days"},
                {"entity_key": "landsat9 swath", "text": "Landsat9 swath is 185 km", "answer": "185 km"},
                {"entity_key": "cryosat2 band", "text": "CryoSat2 band is Ku radar", "answer": "Ku radar"},
                {"entity_key": "gracefo laser", "text": "GraceFO laser is 1064 nm", "answer": "1064 nm"},
            ],
        },
        {
            "domain_id": 1,
            "domain_name": "hydrology",
            "facts": [
                {"entity_key": "danube basin", "text": "Danube basin is 801000 sqkm", "answer": "801000 sqkm"},
                {"entity_key": "aquifer porosity", "text": "Aquifer porosity is 32 percent", "answer": "32 percent"},
                {"entity_key": "snowmelt peak", "text": "Snowmelt peak is May fifteenth", "answer": "May fifteenth"},
                {"entity_key": "weir coefficient", "text": "Weir coefficient is 1.84 metric", "answer": "1.84 metric"},
            ],
        },
        {
            "domain_id": 2,
            "domain_name": "avionics",
            "facts": [
                {"entity_key": "arinc429 rate", "text": "ARINC429 rate is 100 kbps", "answer": "100 kbps"},
                {"entity_key": "pitot heater", "text": "Pitot heater is 28 volts", "answer": "28 volts"},
                {"entity_key": "transponder squawk", "text": "Transponder squawk is 4721 octal", "answer": "4721 octal"},
                {"entity_key": "irs alignment", "text": "IRS alignment is 10 minutes", "answer": "10 minutes"},
            ],
        },
        {
            "domain_id": 3,
            "domain_name": "biochemistry",
            "facts": [
                {"entity_key": "rubisco turnover", "text": "RuBisCO turnover is 3.3 persec", "answer": "3.3 persec"},
                {"entity_key": "atp delta_g", "text": "ATP delta_g is minus 30.5 kJ", "answer": "minus 30.5 kJ"},
                {"entity_key": "heme iron", "text": "Heme iron is ferrous Fe2", "answer": "ferrous Fe2"},
                {"entity_key": "km hexokinase", "text": "Km hexokinase is 0.1 millimolar", "answer": "0.1 millimolar"},
            ],
        },
        {
            "domain_id": 4,
            "domain_name": "cryptography",
            "facts": [
                {"entity_key": "ed25519 pubkey", "text": "Ed25519 pubkey is 32 bytes", "answer": "32 bytes"},
                {"entity_key": "chacha20 rounds", "text": "ChaCha20 rounds is 20 quarter", "answer": "20 quarter"},
                {"entity_key": "blake3 digest", "text": "BLAKE3 digest is 256 bits", "answer": "256 bits"},
                {"entity_key": "kyber768 modulus", "text": "Kyber768 modulus is 3329 prime", "answer": "3329 prime"},
            ],
        },
    ]

    contradictions = [
        {
            "entity_key": "relayuplink frequency",
            "initial_fact": "RelayUplink frequency is 2210 MHz",
            "updated_fact": "RelayUplink frequency is 8425 MHz",
            "query": "relayuplink frequency",
            "expected_new": "8425 MHz",
            "stale_old": "2210 MHz",
        },
        {
            "entity_key": "rover battery limit",
            "initial_fact": "Rover battery limit is 40 percent",
            "updated_fact": "Rover battery limit is 22 percent",
            "query": "rover battery limit",
            "expected_new": "22 percent",
            "stale_old": "40 percent",
        },
        {
            "entity_key": "spectrometer slit width",
            "initial_fact": "Spectrometer slit width is 50 microns",
            "updated_fact": "Spectrometer slit width is 15 microns",
            "query": "spectrometer slit width",
            "expected_new": "15 microns",
            "stale_old": "50 microns",
        },
        {
            "entity_key": "calibration target albedo",
            "initial_fact": "Calibration target albedo is 0.99 spectralon",
            "updated_fact": "Calibration target albedo is 0.52 graypanel",
            "query": "calibration target albedo",
            "expected_new": "0.52 graypanel",
            "stale_old": "0.99 spectralon",
        },
    ]
    return domains, contradictions


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_default_fixtures(fixture_dir: Path = DEFAULT_FIXTURE_DIR) -> Dict[str, Path]:
    """Write deterministic offline JSONL fixtures and verify zero split contamination."""
    fixture_dir = Path(fixture_dir)
    train_docs, val_docs, test_docs = _build_disjoint_prose_corpora()
    decisions = _build_decision_fixtures()
    routing = _build_routing_fixtures()
    domains, contradictions = _build_continual_and_contradiction_fixtures()
    blimp = _build_blimp_probes()

    paths = {
        "corpus_train": fixture_dir / "corpus_train.jsonl",
        "corpus_val_cal": fixture_dir / "corpus_val_cal.jsonl",
        "corpus_test_eval": fixture_dir / "corpus_test_eval.jsonl",
        "blimp_minimal_pairs": fixture_dir / "blimp_minimal_pairs.jsonl",
        "intent_decisions": fixture_dir / "intent_decisions.jsonl",
        "action_routing": fixture_dir / "action_routing.jsonl",
        "continual_domains": fixture_dir / "continual_domains.jsonl",
        "contradictions": fixture_dir / "contradictions.jsonl",
    }
    _write_jsonl(paths["corpus_train"], [{"id": k, "text": v} for k, v in train_docs.items()])
    _write_jsonl(paths["corpus_val_cal"], [{"id": k, "text": v} for k, v in val_docs.items()])
    _write_jsonl(paths["corpus_test_eval"], [{"id": k, "text": v} for k, v in test_docs.items()])
    _write_jsonl(paths["blimp_minimal_pairs"], blimp)
    _write_jsonl(paths["intent_decisions"], decisions)
    _write_jsonl(paths["action_routing"], routing)
    _write_jsonl(paths["continual_domains"], domains)
    _write_jsonl(paths["contradictions"], contradictions)
    return paths


def load_tournament_workloads(
    *,
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    real_data_dir: Optional[Path] = None,
    require_real_data: bool = False,
    scale: str = "fixture",
    mqar_pairs: Sequence[int] = (4, 16, 32, 64),
    mqar_train_per_setting: int = 96,
    mqar_eval_per_setting: int = 32,
    seed: int = 1337,
) -> WorkloadBundle:
    """Load real datasets if provided/available, else deterministic offline fixtures."""
    corpus_metadata: Dict[str, Any] = {}
    if real_data_dir is not None:
        rdir = Path(real_data_dir)
        req_files = [rdir / "train.jsonl", rdir / "val_cal.jsonl", rdir / "test_eval.jsonl"]
        if all(p.is_file() for p in req_files):
            dataset_source = "real"
            train_rows = _read_jsonl(req_files[0])
            val_rows = _read_jsonl(req_files[1])
            test_rows = _read_jsonl(req_files[2])
            lm_train = {r["id"]: r["text"] for r in train_rows}
            lm_val = {r["id"]: r["text"] for r in val_rows}
            lm_test = {r["id"]: r["text"] for r in test_rows}
            prov_file = rdir / "provenance.json"
            if prov_file.is_file():
                corpus_metadata.update(json.loads(prov_file.read_text(encoding="utf-8")))
        elif require_real_data:
            return WorkloadBundle(
                status="blocked",
                dataset_source="missing_real",
                reason=f"Required real dataset files not found in {rdir}",
            )
        else:
            dataset_source = "fixture"
            lm_train, lm_val, lm_test = {}, {}, {}
    elif require_real_data:
        return WorkloadBundle(
            status="blocked",
            dataset_source="missing_real",
            reason="--require-real-data set, but no --real-data-dir supplied in offline sandbox",
        )
    else:
        dataset_source = "fixture"
        lm_train, lm_val, lm_test = {}, {}, {}

    fixture_dir = Path(fixture_dir)
    if not (fixture_dir / "corpus_train.jsonl").is_file():
        write_default_fixtures(fixture_dir)

    if not lm_train:
        lm_train = {r["id"]: r["text"] for r in _read_jsonl(fixture_dir / "corpus_train.jsonl")}
        lm_val = {r["id"]: r["text"] for r in _read_jsonl(fixture_dir / "corpus_val_cal.jsonl")}
        lm_test = {r["id"]: r["text"] for r in _read_jsonl(fixture_dir / "corpus_test_eval.jsonl")}

    # Verify zero contamination across train / val_cal / test_eval
    assert_disjoint_splits(list(lm_val.keys()), list(lm_test.keys()))
    assert_disjoint_splits(list(lm_train.keys()), list(lm_test.keys()))
    train_blob = b"\n".join(v.encode("utf-8") for v in lm_train.values())
    test_bytes_map = {k: v.encode("utf-8") for k, v in lm_test.items()}
    offenders = find_contamination(train_blob, test_bytes_map, window=64, samples_per_document=8)
    if offenders:
        raise ValueError(f"Contamination detected between lm_train and lm_test: {offenders}")

    unigram_stats = compute_corpus_unigram_bpb(lm_train, lm_test)
    corpus_metadata.update(unigram_stats)
    corpus_metadata.setdefault("corpus_name", "Real Corpus" if dataset_source == "real" else "Offline Synthetic Fixture")

    # Extrapolation documents at target byte/token lengths {512, 1024, 2048, 4096}
    base_test_concat = " ".join(lm_test.values())
    extrapolation_docs: Dict[int, List[str]] = {}
    for target_len in (512, 1024, 2048, 4096):
        repeats = max(1, (target_len * 5) // max(1, len(base_test_concat)) + 1)
        long_text = (base_test_concat + " ") * repeats
        extrapolation_docs[target_len] = [long_text[: target_len * 4]]

    is_large = scale.strip().lower() == "large"

    # Grammar probes: repo `evals/grammar_probes.jsonl` + bundled/large BLiMP minimal pairs
    grammar: List[Dict[str, Any]] = []
    if GRAMMAR_PROBES_PATH.is_file():
        grammar.extend(_read_jsonl(GRAMMAR_PROBES_PATH))
    if is_large:
        grammar.extend(build_large_blimp_probes(seed=seed))
    else:
        grammar.extend(_read_jsonl(fixture_dir / "blimp_minimal_pairs.jsonl"))

    eff_mqar_eval = max(256, int(mqar_eval_per_setting)) if is_large else int(mqar_eval_per_setting)
    eff_mqar_train = max(256, int(mqar_train_per_setting)) if is_large else int(mqar_train_per_setting)
    mqar_splits: Dict[int, Dict[str, List[MQARExample]]] = {}
    for kv in mqar_pairs:
        mqar_splits[int(kv)] = {
            "train": generate_mqar_examples(eff_mqar_train, num_kv_pairs=int(kv), seed=seed),
            "eval": generate_mqar_examples(eff_mqar_eval, num_kv_pairs=int(kv), seed=seed + 50_000),
        }
    copy_eval_n = 256 if is_large else 24
    copy_splits = {
        "train": generate_exact_copy_examples(128 if is_large else 64, span_len=12, delay_len=24, seed=seed),
        "eval": generate_exact_copy_examples(copy_eval_n, span_len=12, delay_len=24, seed=seed + 60_000),
    }

    if is_large:
        decisions = build_large_decision_suite(seed=seed)
        routing = build_large_routing_suite(seed=seed)
        domains, contradictions = build_large_continual_suite(seed=seed)
    else:
        dec_rows = _read_jsonl(fixture_dir / "intent_decisions.jsonl")
        decisions = {}
        for r in dec_rows:
            decisions.setdefault(r["split"], []).append(r)
        rt_rows = _read_jsonl(fixture_dir / "action_routing.jsonl")
        routing = {}
        for r in rt_rows:
            routing.setdefault(r["split"], []).append(r)
        domains = _read_jsonl(fixture_dir / "continual_domains.jsonl")
        contradictions = _read_jsonl(fixture_dir / "contradictions.jsonl")

    assert_disjoint_splits(
        [r["id"] for r in decisions.get("val_cal", [])],
        [r["id"] for r in decisions.get("test_in_domain", [])]
        + [r["id"] for r in decisions.get("test_shifted_unseen", [])],
    )
    assert_disjoint_splits(
        [r["id"] for r in routing.get("val_cal", [])],
        [r["id"] for r in routing.get("test_eval", [])],
    )

    mqar_queries_per_setting = sum(len(ex.queries) for ex in mqar_splits[int(mqar_pairs[0])]["eval"])
    item_counts = {
        "scale": scale,
        "lm_train_docs": len(lm_train),
        "lm_val_cal_docs": len(lm_val),
        "lm_test_eval_docs": len(lm_test),
        "lm_train_utf8_bytes": int(unigram_stats["train_utf8_bytes"]),
        "lm_test_eval_utf8_bytes": int(unigram_stats["test_utf8_bytes"]),
        "grammar_probes": len(grammar),
        "mqar_eval_sequences_per_kv": eff_mqar_eval,
        "mqar_eval_queries_per_kv": mqar_queries_per_setting,
        "mqar_eval_queries_total": sum(
            sum(len(ex.queries) for ex in mqar_splits[int(kv)]["eval"]) for kv in mqar_pairs
        ),
        "copy_eval_sequences": copy_eval_n,
        "copy_eval_tokens": copy_eval_n * 12,
        "decision_train": len(decisions.get("train", [])),
        "decision_val_cal": len(decisions.get("val_cal", [])),
        "decision_test_in_domain": len(decisions.get("test_in_domain", [])),
        "decision_test_shifted_unseen": len(decisions.get("test_shifted_unseen", [])),
        "decision_test_total": len(decisions.get("test_in_domain", [])) + len(decisions.get("test_shifted_unseen", [])),
        "routing_train": len(routing.get("train", [])),
        "routing_val_cal": len(routing.get("val_cal", [])),
        "routing_test_eval": len(routing.get("test_eval", [])),
        "continual_domains": len(domains),
        "continual_facts_total": sum(len(d.get("facts", [])) for d in domains),
        "continual_distractors_total": sum(len(d.get("distractors", [])) for d in domains),
        "contradiction_pairs": len(contradictions),
    }

    return WorkloadBundle(
        status="ready",
        dataset_source=dataset_source,
        corpus_metadata=corpus_metadata,
        item_counts=item_counts,
        lm_train=lm_train,
        lm_val_cal=lm_val,
        lm_test_eval=lm_test,
        extrapolation_docs=extrapolation_docs,
        grammar_probes=grammar,
        mqar_splits=mqar_splits,
        copy_splits=copy_splits,
        decisions=decisions,
        routing=routing,
        continual_domains=domains,
        contradiction_pairs=contradictions,
    )
