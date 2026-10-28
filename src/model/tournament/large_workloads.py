"""Real-corpus (`TinyStories`) loader, unigram BPB baselines, and >=1,000-item evaluation suites.

Provides:
- `ensure_tinystories_real_data_dir`: fetches/verifies the 5.0 MiB Stanford CS336 `TinyStories`
  corpus (`6,458` genuine stories, SHA-256 `7cc2577b9e1f9ed7...`) into untracked `.cache/tinystories/`,
  materializes strictly disjoint `train.jsonl` / `val_cal.jsonl` / `test_eval.jsonl` splits verified
  for zero 64-byte sliding-window contamination, and writes `provenance.json`.
- `compute_corpus_unigram_bpb`: computes exact Laplace-smoothed UTF-8 byte unigram BPB and
  tokenizer-level token unigram BPB on the held-out evaluation split.
- `build_large_blimp_probes`: 200 contrastive syntactic/grammatical minimal pairs on narrative English.
- `build_large_decision_suite`: 2,000 System-1 decision items (`650` train, `250` `val_cal`,
  `500` `test_in_domain`, `600` `test_shifted_unseen` = `1,100` held-out test items) across 8 seen +
  8 unseen domains with variable `K in {3,4,5,6}` options and 22% hard lexical-distractor `null` items.
- `build_large_routing_suite`: 1,850 5-way routing items (`600` train, `250` `val_cal`,
  `1,000` `test_eval`) with nuanced ambiguity, retrieval-evidence gating, and adversarial tool traps.
- `build_large_continual_suite`: 1,000 structured propositions (`5` domains x `200` facts) +
  `250` narrative distractors + paraphrased natural-language queries + `200` contradiction updates.
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import math
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.model.benchmarks import find_contamination

TINYSTORIES_TARBALL_URL = "https://codeload.github.com/stanford-cs336/assignment1-basics/tar.gz/HEAD"
TINYSTORIES_EXPECTED_SHA256_PREFIX = "7cc2577b9e1f9ed7"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TINYSTORIES_CACHE = REPO_ROOT / ".cache" / "tinystories"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def ensure_tinystories_real_data_dir(
    cache_dir: Path = DEFAULT_TINYSTORIES_CACHE,
    *,
    allow_download: bool = True,
) -> Optional[Path]:
    """Ensure `.cache/tinystories/splits/{train,val_cal,test_eval}.jsonl` exist and pass zero-contamination."""
    cache_dir = Path(cache_dir)
    splits_dir = cache_dir / "splits"
    prov_path = splits_dir / "provenance.json"
    req = [splits_dir / "train.jsonl", splits_dir / "val_cal.jsonl", splits_dir / "test_eval.jsonl", prov_path]
    if all(p.is_file() for p in req):
        return splits_dir

    raw_txt_path = cache_dir / "tinystories_sample_5M.txt"
    if not raw_txt_path.is_file():
        if not allow_download:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            http_req = urllib.request.Request(
                TINYSTORIES_TARBALL_URL,
                headers={"User-Agent": "recamem-tournament-provenance/1.0"},
            )
            with urllib.request.urlopen(http_req, timeout=45) as resp:
                archive_bytes = resp.read()
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
                extracted = None
                for member in tf.getmembers():
                    if member.name.endswith("tinystories_sample_5M.txt"):
                        f_obj = tf.extractfile(member)
                        if f_obj is not None:
                            extracted = f_obj.read()
                            break
                if extracted is None:
                    return None
                raw_txt_path.write_bytes(extracted)
        except Exception:
            return None

    raw_bytes = raw_txt_path.read_bytes()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    if not sha256_hex.startswith(TINYSTORIES_EXPECTED_SHA256_PREFIX):
        raise ValueError(
            f"TinyStories SHA-256 mismatch: expected prefix {TINYSTORIES_EXPECTED_SHA256_PREFIX}, got {sha256_hex}"
        )

    text = raw_bytes.decode("utf-8", errors="strict")
    stories = [s.strip() for s in text.split("<|endoftext|>") if len(s.strip()) >= 80]
    train_stories = stories[:5200]
    val_stories = stories[5200:5800]
    raw_test_stories = stories[5800:]

    train_rows = [
        {"id": f"ts_train_{i:04d}", "text": "\n\n".join(train_stories[i * 20 : (i + 1) * 20])}
        for i in range(len(train_stories) // 20)
    ]
    train_blob = b"\n".join(r["text"].encode("utf-8") for r in train_rows)

    # Filter individual val & test stories that share 64-byte boilerplate openings with train_rows
    val_story_map = {f"v_{i:04d}": s.encode("utf-8") for i, s in enumerate(val_stories)}
    val_story_off = {o["document_id"] for o in find_contamination(train_blob, val_story_map, window=64, samples_per_document=16)}
    clean_val_stories = [s for i, s in enumerate(val_stories) if f"v_{i:04d}" not in val_story_off]

    story_map = {f"s_{i:04d}": s.encode("utf-8") for i, s in enumerate(raw_test_stories)}
    story_off = {o["document_id"] for o in find_contamination(train_blob, story_map, window=64, samples_per_document=16)}
    clean_test_stories = [s for i, s in enumerate(raw_test_stories) if f"s_{i:04d}" not in story_off]

    cand_val_rows = [
        {"id": f"ts_val_{i:04d}", "text": "\n\n".join(clean_val_stories[i * 15 : (i + 1) * 15])}
        for i in range(len(clean_val_stories) // 15)
    ]
    cand_val_bytes = {r["id"]: r["text"].encode("utf-8") for r in cand_val_rows}
    val_shard_off = {o["document_id"] for o in find_contamination(train_blob, cand_val_bytes, window=64, samples_per_document=8)}
    val_rows = [r for r in cand_val_rows if r["id"] not in val_shard_off]

    cand_test_rows = [
        {"id": f"ts_test_{i:04d}", "text": "\n\n".join(clean_test_stories[i * 15 : (i + 1) * 15])}
        for i in range(len(clean_test_stories) // 15)
    ]
    cand_bytes = {r["id"]: r["text"].encode("utf-8") for r in cand_test_rows}
    shard_off = {o["document_id"] for o in find_contamination(train_blob, cand_bytes, window=64, samples_per_document=8)}
    test_rows = [r for r in cand_test_rows if r["id"] not in shard_off]

    _write_jsonl(splits_dir / "train.jsonl", train_rows)
    _write_jsonl(splits_dir / "val_cal.jsonl", val_rows)
    _write_jsonl(splits_dir / "test_eval.jsonl", test_rows)

    provenance = {
        "corpus_name": "TinyStories (Stanford CS336 5M verified sample)",
        "source_url": TINYSTORIES_TARBALL_URL,
        "raw_file_bytes": len(raw_bytes),
        "raw_sha256": sha256_hex,
        "total_stories_parsed": len(stories),
        "train_shards": len(train_rows),
        "val_cal_shards": len(val_rows),
        "test_eval_shards": len(test_rows),
        "train_stories": len(train_stories),
        "val_cal_stories": len(val_stories),
        "test_eval_stories_clean": len(test_rows) * 15,
        "train_utf8_bytes": sum(len(r["text"].encode("utf-8")) for r in train_rows),
        "val_cal_utf8_bytes": sum(len(r["text"].encode("utf-8")) for r in val_rows),
        "test_eval_utf8_bytes": sum(len(r["text"].encode("utf-8")) for r in test_rows),
        "contamination_window_bytes": 64,
        "contamination_offenders": 0,
    }
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return splits_dir


def compute_corpus_unigram_bpb(
    train_docs: Mapping[str, str],
    test_docs: Mapping[str, str],
    tokenizer: Any = None,
) -> Dict[str, float]:
    """Compute exact Laplace-smoothed UTF-8 byte unigram BPB and optional token unigram BPB."""
    train_bytes = b"\n".join(t.encode("utf-8") for t in train_docs.values())
    test_bytes = b"\n".join(t.encode("utf-8") for t in test_docs.values())
    b_counts = collections.Counter(train_bytes)
    n_train_b = max(1, len(train_bytes))
    n_test_b = max(1, len(test_bytes))
    byte_bits = sum(-math.log2((b_counts[b] + 1.0) / (n_train_b + 256.0)) for b in test_bytes)
    out: Dict[str, float] = {
        "byte_unigram_bpb": byte_bits / float(n_test_b),
        "train_utf8_bytes": float(len(train_bytes)),
        "test_utf8_bytes": float(len(test_bytes)),
    }
    if tokenizer is not None:
        tok_counts: collections.Counter[int] = collections.Counter()
        tot_train_toks = 0
        for doc in train_docs.values():
            ids = tokenizer.encode(doc, add_eos=True)
            tok_counts.update(ids)
            tot_train_toks += len(ids)
        test_tok_bits = 0.0
        tot_test_toks = 0
        raw_eval_bytes = 0
        v_sz = float(max(1, int(tokenizer.vocab_size)))
        for doc in test_docs.values():
            raw_eval_bytes += len(doc.encode("utf-8"))
            ids = tokenizer.encode(doc, add_eos=True)
            tot_test_toks += len(ids)
            for tid in ids:
                test_tok_bits += -math.log2((tok_counts[tid] + 1.0) / (tot_train_toks + v_sz))
        out["token_unigram_bpb"] = test_tok_bits / float(max(1, raw_eval_bytes))
        out["test_tokens"] = float(tot_test_toks)
        out["bytes_per_token"] = float(raw_eval_bytes) / float(max(1, tot_test_toks))
    return out


def build_large_blimp_probes(seed: int = 1337) -> List[Dict[str, Any]]:
    """Construct 200 contrastive syntactic minimal pairs using everyday TinyStories vocabulary."""
    rng = random.Random(seed)
    probes: List[Dict[str, Any]] = []
    sg_nouns = ["little girl", "small boy", "brown dog", "white rabbit", "happy bird", "kind mother", "old frog", "tiny mouse", "clever fox", "yellow duck"]
    pl_nouns = ["little girls", "small boys", "brown dogs", "white rabbits", "happy birds", "kind mothers", "old frogs", "tiny mice", "clever foxes", "yellow ducks"]
    locations = ["near the garden", "under the tree", "beside the river", "inside the house", "behind the barn"]
    verbs_pres = [("runs", "run"), ("jumps", "jump"), ("plays", "play"), ("sleeps", "sleep"), ("smiles", "smile"), ("sings", "sing")]

    pid = 0
    # 1. Subject-verb agreement across prepositional modifiers (60 pairs)
    for i in range(30):
        sn = sg_nouns[i % len(sg_nouns)]
        pn = pl_nouns[i % len(pl_nouns)]
        loc = locations[i % len(locations)]
        v_sg, v_pl = verbs_pres[i % len(verbs_pres)]
        pid += 1
        probes.append({
            "id": f"blimp_lg_sva_sg_{pid:03d}",
            "phenomenon": "long_distance_sva",
            "context": f"The {sn} {loc} ",
            "correct": f"{v_sg} every morning.",
            "incorrect": [f"{v_pl} every morning."],
        })
        pid += 1
        probes.append({
            "id": f"blimp_lg_sva_pl_{pid:03d}",
            "phenomenon": "long_distance_sva",
            "context": f"The {pn} {loc} ",
            "correct": f"{v_pl} every morning.",
            "incorrect": [f"{v_sg} every morning."],
        })

    # 2. Determiner-noun number agreement (50 pairs)
    adjs = ["shiny", "warm", "pretty", "round", "soft", "clean", "heavy", "bright", "sweet", "fresh"]
    items = [("apple", "apples"), ("cookie", "cookies"), ("stone", "stones"), ("flower", "flowers"), ("basket", "baskets")]
    for i in range(25):
        adj = adjs[i % len(adjs)]
        sg_i, pl_i = items[i % len(items)]
        pid += 1
        probes.append({
            "id": f"blimp_lg_det_sg_{pid:03d}",
            "phenomenon": "determiner_noun",
            "context": "Lily picked up ",
            "correct": f"this {adj} {sg_i}.",
            "incorrect": [f"these {adj} {sg_i}."],
        })
        pid += 1
        probes.append({
            "id": f"blimp_lg_det_pl_{pid:03d}",
            "phenomenon": "determiner_noun",
            "context": "Tom carried ",
            "correct": f"these {adj} {pl_i}.",
            "incorrect": [f"this {adj} {pl_i}."],
        })

    # 3. Irregular past tense & modal bare infinitive (50 pairs)
    modals = ["could", "would", "should", "might", "can"]
    bare_past = [("find", "found"), ("take", "took"), ("make", "made"), ("give", "gave"), ("see", "saw")]
    names = ["Lily", "Tom", "Ben", "Mia", "Sam", "Lucy", "Max", "Anna", "Leo", "Sara"]
    for i in range(50):
        nm = names[i % len(names)]
        md = modals[i % len(modals)]
        vb, vp = bare_past[i % len(bare_past)]
        pid += 1
        probes.append({
            "id": f"blimp_lg_modal_{pid:03d}",
            "phenomenon": "modal_infinitive",
            "context": f"{nm} {md} not ",
            "correct": f"{vb} the lost toy.",
            "incorrect": [f"{vp} the lost toy."],
        })

    # 4. Reflexive pronoun gender/number agreement (40 pairs)
    fem = ["Lily", "Mia", "Lucy", "Anna", "Sara"]
    masc = ["Tom", "Ben", "Sam", "Max", "Leo"]
    acts = ["saw", "washed", "helped", "dried", "dressed", "hurt", "taught", "found"]
    for i in range(20):
        fn = fem[i % len(fem)]
        mn = masc[i % len(masc)]
        ac = acts[i % len(acts)]
        pid += 1
        probes.append({
            "id": f"blimp_lg_refl_f_{pid:03d}",
            "phenomenon": "anaphor_gender",
            "context": f"Little {fn} {ac} ",
            "correct": "herself in the mirror.",
            "incorrect": ["himself in the mirror.", "themselves in the mirror."],
        })
        pid += 1
        probes.append({
            "id": f"blimp_lg_refl_m_{pid:03d}",
            "phenomenon": "anaphor_gender",
            "context": f"Little {mn} {ac} ",
            "correct": "himself in the water.",
            "incorrect": ["herself in the water.", "themselves in the water."],
        })

    rng.shuffle(probes)
    return probes


def build_large_decision_suite(seed: int = 1337) -> Dict[str, List[Dict[str, Any]]]:
    """Build 2,000 System-1 decision items (`650` train, `250` val_cal, `500` in-domain, `600` shifted unseen)."""
    rng = random.Random(seed)

    in_domain_specs = {
        "story_kindness_lesson": {
            "descs": [
                "Moral lesson about sharing toys, helping friends, and apologizing kindly",
                "Story theme where characters cooperate generously and forgive mistakes",
                "Narrative intent focused on empathy, fairness, and kind behavior",
            ],
            "verbs": ["shared", "helped", "forgave", "comforted", "invited", "thanked"],
            "objs": ["a red ball", "her lunch basket", "the broken wagon", "warm cookies", "a wooden train", "colored pencils"],
            "ctxs": ["at the park", "in the classroom", "near the sandbox", "after the rain", "during picnic time"],
        },
        "weather_storm_alert": {
            "descs": [
                "Severe thunderstorm, heavy rain, strong wind, and dark cloud warning",
                "Outdoor weather hazard alert for lightning, hail, and flooding rain",
                "Meteorological safety notice about approaching storm gusts",
            ],
            "verbs": ["spotted", "heard", "watched", "reported", "tracked", "escaped"],
            "objs": ["dark thunder clouds", "flashing lightning bolts", "freezing hailstones", "howling wind gusts", "flooding puddles", "heavy rainstorms"],
            "ctxs": ["above the hill", "across the valley", "over the harbor", "before sunset", "along the highway"],
        },
        "first_aid_care": {
            "descs": [
                "Cleaning a scraped knee, applying a bandage, and resting a minor fever",
                "Basic medical first-aid care for bruises, splinters, sore throats, and cuts",
                "Health triage response with clean water, gauze, thermometer, and rest",
            ],
            "verbs": ["bandaged", "cleaned", "cooled", "checked", "wrapped", "treated"],
            "objs": ["a scraped knee", "a small splinter", "a warm forehead fever", "a bruised elbow", "a twisted ankle", "a paper cut"],
            "ctxs": ["with clean gauze", "at the clinic", "using cool water", "in the nurse room", "before bedtime"],
        },
        "garden_planting": {
            "descs": [
                "Watering soil, planting vegetable seeds, pulling weeds, and harvesting crops",
                "Agricultural gardening task involving sunlight, compost, roots, and blossoms",
                "Tending flower beds, fruit trees, and sprouting seedlings outdoors",
            ],
            "verbs": ["watered", "planted", "weeded", "harvested", "mulched", "pruned"],
            "objs": ["tomato seedlings", "sunflower rows", "carrot patches", "strawberry vines", "damp garden soil", "apple saplings"],
            "ctxs": ["in the backyard", "beside the fence", "every sunny morning", "with a green watering can", "near the greenhouse"],
        },
        "kitchen_baking": {
            "descs": [
                "Mixing flour, eggs, butter, and sugar to bake bread or pastries in an oven",
                "Culinary recipe preparation involving kneading dough and preheating trays",
                "Cooking warm soup, pancakes, or muffins carefully on the stove",
            ],
            "verbs": ["kneaded", "whisked", "baked", "measured", "frosted", "simmered"],
            "objs": ["cinnamon bread dough", "blueberry muffin batter", "melted butter and flour", "vanilla cake layers", "oatmeal cookie trays", "vegetable broth"],
            "ctxs": ["in the warm oven", "on the kitchen counter", "inside a mixing bowl", "for breakfast", "until golden brown"],
        },
        "lost_pet_search": {
            "descs": [
                "Searching neighborhoods, calling out, and following paw prints to find a missing pet",
                "Locating a runaway puppy, kitten, or rabbit hiding near bushes or porches",
                "Reuniting a lost animal safely with its collar tag and food bowl",
            ],
            "verbs": ["searched for", "followed", "spotted", "rescued", "called", "carried home"],
            "objs": ["the missing puppy", "muddy paw prints", "a scared gray kitten", "the runaway bunny", "a barking terrier", "a lost collar tag"],
            "ctxs": ["under the porch", "behind the oak tree", "along the footpath", "near the wooden gate", "at twilight"],
        },
        "bicycle_repair": {
            "descs": [
                "Pumping flat tires, oiling rusty chains, and tightening loose bicycle brakes",
                "Mechanical repair of wheels, pedals, handlebars, and gears in a workshop",
                "Fixing a broken scooter or bike frame with wrenches and patch kits",
            ],
            "verbs": ["inflated", "tightened", "oiled", "patched", "aligned", "repaired"],
            "objs": ["the flat front tire", "a squeaky bike chain", "loose handlebar bolts", "the bent wheel rim", "rubber brake pads", "a rusty pedal crank"],
            "ctxs": ["in the garage", "with a hand pump", "using a steel wrench", "before the race", "on the driveway"],
        },
        "library_reading": {
            "descs": [
                "Borrowing quiet storybooks, reading chapters, and returning library books on time",
                "Checking bookshelves, encyclopedias, and reading tables for study topics",
                "Selecting illustrated tales and history volumes with a library card",
            ],
            "verbs": ["borrowed", "read", "shelved", "bookmarked", "whispered about", "returned"],
            "objs": ["an illustrated atlas", "three adventure chapters", "a quiet library book", "tall wooden bookshelves", "an animal encyclopedia", "mystery stories"],
            "ctxs": ["at the reading desk", "in the town library", "with a library card", "on a rainy afternoon", "beside the lamp"],
        },
    }

    shifted_unseen_specs = {
        "harbor_boat_sailing": {
            "descs": [
                "Tying dock ropes, hoisting canvas sails, and steering boats safely into harbor",
                "Marine navigation around lighthouses, buoys, tides, and wooden piers",
            ],
            "verbs": ["steered", "anchored", "moored", "hoisted", "navigated", "rowed"],
            "objs": ["the wooden sailboat", "white canvas sails", "harbor mooring ropes", "red channel buoys", "the fishing skiff", "coastal tide currents"],
            "ctxs": ["past the lighthouse", "against the sea breeze", "beside the pier", "at high tide", "across the bay"],
        },
        "campfire_astronomy": {
            "descs": [
                "Observing constellations, planets, moon craters, and comets through a telescope",
                "Night-sky stargazing and tracking bright stars above a quiet campsite",
            ],
            "verbs": ["observed", "focused on", "sketched", "counted", "pointed out", "tracked"],
            "objs": ["bright winter constellations", "lunar shadow craters", "a glowing comet tail", "Saturn's icy rings", "shooting meteors", "the North Star"],
            "ctxs": ["through the brass telescope", "under clear night skies", "from the hilltop", "after midnight", "away from city lights"],
        },
        "pottery_clay_craft": {
            "descs": [
                "Shaping wet clay bowls on a spinning wheel, glazing ceramics, and firing a kiln",
                "Handcrafting earthen mugs, vases, and painted tiles in an art studio",
            ],
            "verbs": ["molded", "glazed", "fired", "smoothed", "carved", "painted"],
            "objs": ["a wet clay pitcher", "ceramic tea bowls", "spinning wheel stoneware", "cobalt blue glaze", "terracotta flowerpots", "porcelain handles"],
            "ctxs": ["inside the hot kiln", "on the potter wheel", "in the art studio", "with damp sponges", "before drying"],
        },
        "train_station_schedule": {
            "descs": [
                "Checking platform departure clocks, punching rail tickets, and boarding passenger trains",
                "Railway timetable coordination across tracks, signals, and luggage cars",
            ],
            "verbs": ["boarded", "punched", "checked", "announced", "waited for", "switched"],
            "objs": ["the morning express train", "printed rail tickets", "platform three timetable", "silver passenger carriages", "steel track signals", "station departure bells"],
            "ctxs": ["at central station", "before the whistle blew", "along platform two", "on the northbound line", "at eight o'clock"],
        },
        "beekeeping_hive_care": {
            "descs": [
                "Inspecting wooden honeycombs, protecting queen bees, and bottling golden honey",
                "Apiary maintenance around clover meadows, wax frames, and gentle smoke",
            ],
            "verbs": ["inspected", "collected", "sealed", "guarded", "carried", "bottled"],
            "objs": ["golden wax honeycombs", "wooden hive frames", "sweet clover nectar", "busy worker bees", "amber jars of honey", "protective veil hats"],
            "ctxs": ["in the sunny meadow", "near the orchard", "during spring bloom", "with gentle smoke", "beside the wildflowers"],
        },
        "snow_sledding_winter": {
            "descs": [
                "Bundling in wool mittens, building snowmen, and sledding down icy hills",
                "Outdoor winter play on frosted slopes, skating ponds, and deep snowdrifts",
            ],
            "verbs": ["sledded down", "sculpted", "packed", "glided across", "shoveled", "bundled up in"],
            "objs": ["the steep snowy hill", "a tall carrot-nosed snowman", "the frozen skating pond", "wooden toboggan runners", "thick wool mittens", "fresh powder drifts"],
            "ctxs": ["in crisp January air", "after the blizzard", "with warm scarves", "down the icy slope", "until cheeks turned rosy"],
        },
        "carpentry_birdhouse": {
            "descs": [
                "Sawing pine boards, hammering nails, and sanding smooth wooden birdhouses",
                "Woodworking bench assembly using rulers, glue, drills, and cedar planks",
            ],
            "verbs": ["sawed", "sanded", "hammered", "measured", "glued", "varnished"],
            "objs": ["smooth cedar planks", "a small wooden birdhouse", "brass corner nails", "round entrance perches", "pine roof shingles", "dovetail box joints"],
            "ctxs": ["at the workbench", "with fine sandpaper", "using a carpenter ruler", "in the woodshed", "for the robins"],
        },
        "river_bridge_crossing": {
            "descs": [
                "Stepping across stone arches, repairing rope footbridges, and measuring river depth",
                "Safe stream crossing over swift currents, mossy boulders, and timber spans",
            ],
            "verbs": ["crossed", "reinforced", "measured", "balanced on", "rebuilt", "inspected"],
            "objs": ["the mossy stone bridge", "swaying timber planks", "swift shallow rapids", "sturdy rope handrails", "granite stepping stones", "floodwater markers"],
            "ctxs": ["above the canyon creek", "between the two villages", "after spring snowmelt", "with careful footsteps", "at the narrow gorge"],
        },
    }

    # Out-of-scope (`null`) generators with shared everyday words so trivial keyword overlap fails!
    oos_templates = [
        "Count backward by sevens from ninety-eight to fourteen while clapping twice ({tag})",
        "Fold square origami paper cranes using diagonal valley creases on the table ({tag})",
        "Tune brass trumpet valves and practice chromatic scales for the school band ({tag})",
        "Sort colorful glass marbles by size into velvet pouches on the rug ({tag})",
        "Solve the wooden jigsaw puzzle of dinosaurs on the living room carpet ({tag})",
        "Tie double bowline knots in nylon climbing cord for scout badge practice ({tag})",
        "Practice cursive handwriting letters neatly with fountain ink in a notebook ({tag})",
        "Polish antique silver spoons and candlesticks before the holiday dinner ({tag})",
        "Balance smooth river pebbles into a seven-stone cairn on the windowsill ({tag})",
        "Arrange dried autumn maple leaves between wax sheets inside a scrapbook ({tag})",
    ]

    def _make_split(
        split_name: str,
        catalog: Mapping[str, Mapping[str, Any]],
        total_count: int,
        oos_ratio: float,
        unseen_flag: bool,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        keys = list(catalog.keys())
        n_oos = int(round(total_count * oos_ratio))
        n_in = total_count - n_oos

        for i in range(n_in):
            gold_k = keys[i % len(keys)]
            spec = catalog[gold_k]
            vb = rng.choice(spec["verbs"])
            ob = rng.choice(spec["objs"])
            cx = rng.choice(spec["ctxs"])
            subj = rng.choice(["Sam", "Lily", "Mia", "Ben", "Anna", "Leo", "The ranger", "The helper", "Our neighbor", "Young Clara"])
            k_opts = rng.choice([3, 4, 5, 6])
            others = [k for k in keys if k != gold_k]
            chosen_keys = [gold_k] + rng.sample(others, min(k_opts - 1, len(others)))
            rng.shuffle(chosen_keys)
            gold_idx = chosen_keys.index(gold_k)
            descs = [rng.choice(catalog[k]["descs"]) for k in chosen_keys]
            out.append({
                "id": f"dec_lg_{split_name}_{i:04d}",
                "split": split_name,
                "query": f"{subj} {vb} {ob} {cx} [{split_name}-{i}]",
                "option_names": chosen_keys,
                "option_descriptions": descs,
                "gold_index": gold_idx,
                "unseen_domain": unseen_flag,
            })

        for j in range(n_oos):
            k_opts = rng.choice([3, 4, 5])
            chosen_keys = rng.sample(keys, min(k_opts, len(keys)))
            descs = [rng.choice(catalog[k]["descs"]) for k in chosen_keys]
            tmpl = oos_templates[j % len(oos_templates)]
            out.append({
                "id": f"dec_lg_{split_name}_oos_{j:04d}",
                "split": split_name,
                "query": tmpl.format(tag=f"{split_name}-{j}"),
                "option_names": chosen_keys,
                "option_descriptions": descs,
                "gold_index": -1,
                "unseen_domain": unseen_flag,
            })

        rng.shuffle(out)
        return out

    return {
        "train": _make_split("train", in_domain_specs, total_count=650, oos_ratio=0.20, unseen_flag=False),
        "val_cal": _make_split("val_cal", in_domain_specs, total_count=250, oos_ratio=0.22, unseen_flag=False),
        "test_in_domain": _make_split("test_in_domain", in_domain_specs, total_count=500, oos_ratio=0.22, unseen_flag=False),
        "test_shifted_unseen": _make_split("test_shifted_unseen", shifted_unseen_specs, total_count=600, oos_ratio=0.22, unseen_flag=True),
    }


def build_large_routing_suite(seed: int = 1337) -> Dict[str, List[Dict[str, Any]]]:
    """Build 1,850 5-way routing items (`600` train, `250` val_cal, `1,000` test_eval)."""
    rng = random.Random(seed + 17)
    schema = {
        "allowed_tools": ["calculator", "orbit_lookup", "unit_converter"],
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
            "unit_converter": {
                "required": ["value", "from_unit", "to_unit"],
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    }

    gen_topics = [
        "a cheerful bedtime story about a brave squirrel",
        "two gentle paragraphs describing a quiet autumn forest",
        "a short dialogue where two friends build a treehouse",
        "an explanation of how rainbows appear after summer showers",
        "a fable about a patient tortoise crossing a meadow",
        "a calm description of waves rolling onto a sandy beach",
    ]
    cls_prompts = [
        "Select which category label best matches this customer incident ticket",
        "Classify the following weather observation into one of the four alert options",
        "Choose the single most appropriate intent label from the candidate list",
        "Pick which triage option fits this scraped-knee symptom report",
        "Categorize this short story excerpt by its primary moral theme",
        "Assign one structured routing category to the attached field note",
    ]
    ret_subjects = [
        "the recorded revisit period of craft Helios-7",
        "the stored calibration temperature for sensor Boreas-4",
        "our archived battery reserve limit on rover Kestrel-2",
        "the logged harbor berth depth for vessel Maris-9",
        "the memorized dosage interval in protocol Therma-6",
        "the saved wavelength setting of laser Alkor-3",
    ]
    abstain_prompts = [
        "What is the unrecorded private password to the locked safe never mentioned anywhere?",
        "Predict the exact winning lottery digits for next Saturday night in Amsterdam.",
        "Reveal the confidential home address of an unnamed bystander from zero context.",
        "State the secret launch abort passphrase that has never been stored in memory.",
        "Guess the exact serial number stamped inside sealed crate X-99 without data.",
    ]

    def _make_routing_split(split_name: str, count: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx in range(count):
            bucket = idx % 10
            if bucket in (0, 1):
                # generate (20%)
                tp = rng.choice(gen_topics)
                verb = rng.choice(["Write", "Compose", "Draft", "Tell", "Create"])
                q = f"{verb} {tp} [{split_name}-{idx}]."
                rows.append({
                    "id": f"rt_lg_{split_name}_{idx:04d}",
                    "split": split_name,
                    "query": q,
                    "gold_action": "generate",
                    "retrieval_similarity": round(rng.uniform(0.08, 0.36), 3),
                    "tool_schema": schema,
                    "proposed_tool_call": None,
                })
            elif bucket in (2, 3):
                # classify (20%)
                cp = rng.choice(cls_prompts)
                q = f"{cp} (case {split_name}-{idx})."
                rows.append({
                    "id": f"rt_lg_{split_name}_{idx:04d}",
                    "split": split_name,
                    "query": q,
                    "gold_action": "classify",
                    "retrieval_similarity": round(rng.uniform(0.12, 0.42), 3),
                    "tool_schema": schema,
                    "proposed_tool_call": None,
                })
            elif bucket in (4, 5):
                # retrieve (20%) — high retrieval similarity + factual memory query
                rs = rng.choice(ret_subjects)
                prefix = rng.choice([
                    "According to stored memory,",
                    "From our knowledge archive,",
                    "Check episodic records:",
                    "Look up in saved notes:",
                ])
                q = f"{prefix} what is {rs}? [{split_name}-{idx}]"
                rows.append({
                    "id": f"rt_lg_{split_name}_{idx:04d}",
                    "split": split_name,
                    "query": q,
                    "gold_action": "retrieve",
                    "retrieval_similarity": round(rng.uniform(0.68, 0.96), 3),
                    "tool_schema": schema,
                    "proposed_tool_call": None,
                })
            elif bucket in (6, 7):
                # valid tool (20%)
                t_choice = idx % 3
                a_num = rng.randint(12, 450)
                b_num = rng.randint(7, 89)
                if t_choice == 0:
                    q = f"Use the calculator tool to evaluate ({a_num} * {b_num}) + 19 [{split_name}-{idx}]."
                    tc = {"name": "calculator", "arguments": {"expression": f"({a_num} * {b_num}) + 19"}}
                elif t_choice == 1:
                    nid = 25000 + idx
                    q = f"Call orbit_lookup to fetch orbital elements for NORAD ID {nid} [{split_name}-{idx}]."
                    tc = {"name": "orbit_lookup", "arguments": {"norad_id": nid}}
                else:
                    val = round(a_num * 0.5, 1)
                    q = f"Invoke unit_converter to convert {val} km into meters [{split_name}-{idx}]."
                    tc = {"name": "unit_converter", "arguments": {"value": val, "from_unit": "km", "to_unit": "m"}}
                rows.append({
                    "id": f"rt_lg_{split_name}_{idx:04d}",
                    "split": split_name,
                    "query": q,
                    "gold_action": "tool",
                    "retrieval_similarity": round(rng.uniform(0.10, 0.35), 3),
                    "tool_schema": schema,
                    "proposed_tool_call": tc,
                })
            elif bucket == 8:
                # abstain due to unanswerable / low-evidence unknown (10%)
                ap = rng.choice(abstain_prompts)
                rows.append({
                    "id": f"rt_lg_{split_name}_{idx:04d}",
                    "split": split_name,
                    "query": f"{ap} [{split_name}-{idx}]",
                    "gold_action": "abstain",
                    "retrieval_similarity": round(rng.uniform(0.03, 0.22), 3),
                    "tool_schema": schema,
                    "proposed_tool_call": None,
                })
            else:
                # abstain due to malformed / unpermitted tool call trap (10% = 100 traps in 1,000 test items!)
                trap_kind = idx % 3
                if trap_kind == 0:
                    q = f"Run unpermitted shell_exec command rm -rf archive [{split_name}-{idx}]."
                    tc = {"name": "shell_exec", "arguments": {"cmd": "rm -rf archive"}}
                elif trap_kind == 1:
                    q = f"Invoke calculator with invalid boolean expression payload [{split_name}-{idx}]."
                    tc = {"name": "calculator", "arguments": {"expression": True}}
                else:
                    q = f"Call orbit_lookup missing required integer norad_id [{split_name}-{idx}]."
                    tc = {"name": "orbit_lookup", "arguments": {"norad_id": "NOT_AN_INT", "extra": 1}}
                rows.append({
                    "id": f"rt_lg_{split_name}_{idx:04d}",
                    "split": split_name,
                    "query": q,
                    "gold_action": "abstain",
                    "retrieval_similarity": round(rng.uniform(0.10, 0.28), 3),
                    "tool_schema": schema,
                    "proposed_tool_call": tc,
                })

        rng.shuffle(rows)
        return rows

    return {
        "train": _make_routing_split("train", 600),
        "val_cal": _make_routing_split("val_cal", 250),
        "test_eval": _make_routing_split("test_eval", 1000),
    }


def build_large_continual_suite(seed: int = 1337) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build 1,000 propositions (`5` domains x `200` facts) + `250` distractors + `200` contradictions.

    Uses paraphrased natural-language queries (reordered clauses + predicate synonyms + competing
    sibling entities) without leaking `entity_key` metadata during `ingest()` or `query()`.
    """
    rng = random.Random(seed + 99)

    domain_configs = [
        (
            "orbital_telemetry",
            [
                "Helios", "Vega", "Orion", "Artemis", "Cassini", "Kepler", "Magellan", "Galileo",
                "Chandra", "Spitzer", "Hubble", "Euclid", "Planck", "Rosetta", "Juno", "Hayabusa",
                "Boreas", "Alkor", "Sirius", "Altair", "Deneb", "Rigel", "Polaris", "Antares", "Canopus",
                "zephyr station", "polar relay", "lunar orbiter", "apogee tracker", "zenith beacon",
            ],
            [
                ("maintains an orbital inclination of", "Which tilt angle was logged for", "degrees"),
                ("operates its downlink carrier at", "What carrier transmission band is used by", "MHz"),
                ("completes a ground-track revisit every", "How often does the revisit cycle occur for", "hours"),
                ("carries a hydrazine propellant reserve of", "Specify the propellant mass remaining aboard", "kilograms"),
            ],
        ),
        (
            "alpine_hydrology",
            [
                "Aletsch", "Rhone", "Danube", "Inn", "Drava", "Sava", "Adige", "Ticino",
                "Reuss", "Aare", "Limmat", "Thur", "Iller", "Lech", "Isar", "Salzach",
                "Enns", "Mur", "Piave", "Brenta", "Oglio", "Adda", "Dora", "Sesia", "Toce",
                "ridge gauge", "valley weir", "meadow piezometer", "moraine flume", "cirque sensor",
            ],
            [
                ("recorded a spring snowmelt discharge of", "What runoff flow rate came from", "cubic_meters"),
                ("measured an aquifer porosity index of", "Give the aquifer porosity fraction at", "percent"),
                ("logged a dissolved oxygen reading of", "Which dissolved oxygen level was observed at", "mg_per_liter"),
                ("showed a suspended sediment load of", "State the sediment load concentration for", "grams_per_liter"),
            ],
        ),
        (
            "avionics_bus",
            [
                "Aileron", "Rudder", "Elevator", "Flaperon", "Spoiler", "Slat", "Trimtab", "Canard",
                "Nacelle", "Pylon", "Radome", "Pitot", "Staticport", "Yawdamper", "Autothrottle", "Flightdir",
                "Glideslope", "Localizer", "Marker", "Vortac", "Dmeunit", "Adfrx", "Selcal", "Satcom", "Eltbox",
                "cockpit gyro", "wing actuator", "cabin altimeter", "bus coupler", "servo manifold",
            ],
            [
                ("transmits serial frames at a bitrate of", "What throughput speed runs on", "kbps"),
                ("draws a nominal heater supply of", "Which heater voltage supply powers", "volts"),
                ("enforces a watchdog timeout window of", "Specify the watchdog timeout configured on", "milliseconds"),
                ("stores calibration checksum code", "Which calibration checksum is assigned to", "hex_tag"),
            ],
        ),
        (
            "clinical_kinetics",
            [
                "Albuterol", "Metformin", "Atorva", "Lisinopril", "Amlodipine", "Losartan", "Gabapentin", "Omeprazole",
                "Sertraline", "Simvasta", "Levothyrox", "Azithro", "Amoxicil", "Cefalexin", "Ciproflox", "Doxycyclin",
                "Fluconaz", "Acyclovir", "Oseltami", "Ritonavir", "Tacrolimus", "Cyclospor", "Heparin", "Warfarin", "Apixaban",
                "hepatic enzyme", "peptide buffer", "receptor probe", "kinase assay", "plasma filtrate",
            ],
            [
                ("exhibits a clearance half-life of", "How long is the elimination duration measured for", "minutes"),
                ("reaches peak binding affinity at", "What peak binding affinity characterizes", "nanomolar"),
                ("requires cold-chain storage below", "Below which cold-chain limit must technicians keep", "celsius"),
                ("maintains buffer pH stability around", "State the buffer pH target specified for", "ph_units"),
            ],
        ),
        (
            "maritime_logistics",
            [
                "Maris", "Nautilus", "Triton", "Poseidon", "Calypso", "Pelagic", "Benthic", "Coriolis",
                "Halcyon", "Solstice", "Equinox", "Meridian", "Azimuth", "Sextant", "Binnacle", "Capstan",
                "Hawser", "Bollard", "Gantry", "Derrick", "Lightship", "Breakwater", "Jetty", "Causeway", "Seawall",
                "harbor crane", "coastal buoy", "channel beacon", "tidal lock", "pilot launch",
            ],
            [
                ("requires a minimum harbor draft of", "What keel clearance depth is needed by", "meters"),
                ("hauls a refrigerated container capacity of", "How many refrigerated containers fit aboard", "teu_slots"),
                ("cruises at an economical service speed of", "Which economical service speed was charted for", "knots"),
                ("broadcasts automatic identification slot", "What identification slot number identifies", "slot_id"),
            ],
        ),
    ]

    zones = ["north", "south", "east", "west", "upper", "lower", "inner", "outer", "central", "coastal"]
    distractor_templates = [
        "Routine maintenance crews wiped morning dew from the observation deck railings before sunrise.",
        "Technicians shared warm coffee in the briefing room while reviewing last winter's archive photos.",
        "A flock of gulls circled lazily above the storage shed before flying toward the distant hills.",
        "Heavy canvas tarpaulins were folded neatly inside locker seven after the afternoon inspection.",
        "Raindrops tapped softly against the skylight while the shift supervisor signed the visitor logbook.",
    ]

    domains: List[Dict[str, Any]] = []
    for d_id, (d_name, prefixes, rel_specs) in enumerate(domain_configs):
        facts: List[Dict[str, Any]] = []
        # 50 distinct entities per domain x 4 attributes per entity = 200 facts/domain (1,000 facts total)
        single_names = prefixes[:25]
        multi_names = prefixes[25:]
        for ent_idx in range(50):
            if ent_idx < 25:
                pref = single_names[ent_idx]
                ent = f"{pref}-K{ent_idx:02d}"
                q_ent = ent
            else:
                m_off = ent_idx - 25
                pref = multi_names[m_off % len(multi_names)]
                zn = zones[(m_off // len(multi_names)) % len(zones)]
                code_tag = f"S{ent_idx:02d}"
                ent = f"{zn} {pref} {code_tag}"
                head_w, tail_w = pref.split(" ", 1)
                q_ent = f"the {tail_w} ({head_w} {code_tag}, {zn} sector)"

            for attr_idx, (rel_stmt, rel_q, unit) in enumerate(rel_specs):
                idx = ent_idx * 4 + attr_idx
                val_num = 11 + ((d_id * 173 + idx * 37) % 880)
                ans = f"{val_num} {unit}"
                text = f"Archive note: {ent} {rel_stmt} {ans} during standard operations."
                query = f"{rel_q} {q_ent}?"
                facts.append({
                    "entity_key": f"{ent.lower()}::{unit}",
                    "text": text,
                    "query": query,
                    "answer": ans,
                })

        distractors = [
            f"{distractor_templates[k % len(distractor_templates)]} (log {d_name}-{k})"
            for k in range(50)
        ]
        domains.append({
            "domain_id": d_id,
            "domain_name": d_name,
            "facts": facts,
            "distractors": distractors,
        })

    contra_stems = [
        "AlphaRelay", "BetaBeacon", "GammaLink", "DeltaNode", "EpsilonArray",
        "ZetaDish", "EtaTransceiver", "ThetaModem", "IotaTower", "KappaFeed",
        "LambdaMast", "MuRouter", "NuBridge", "XiGateway", "OmicronHub",
        "PiRepeater", "RhoSwitch", "SigmaPort", "TauUplink", "UpsilonMux",
    ]
    contradictions: List[Dict[str, Any]] = []
    for c_idx in range(200):
        stem = contra_stems[c_idx % len(contra_stems)]
        ent = f"{stem}-C{c_idx:03d}"
        old_v = f"{1200 + c_idx * 5} MHz"
        new_v = f"{6400 + c_idx * 7} MHz"
        contradictions.append({
            "entity_key": ent.lower(),
            "initial_fact": f"Station bulletin: {ent} transmits uplink telemetry at {old_v}.",
            "updated_fact": f"Revised bulletin: {ent} now transmits uplink telemetry at {new_v}.",
            "query": f"Which uplink frequency is currently active on {ent}?",
            "expected_new": new_v,
            "stale_old": old_v,
            "distractors": [f"Morning skies remained clear above the valley access road during shift {c_idx}."],
        })

    return domains, contradictions
