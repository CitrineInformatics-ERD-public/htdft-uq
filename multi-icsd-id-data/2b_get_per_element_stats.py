import os
import sys
import json
import gzip
import warnings
import time
import itertools as it
from collections import Counter
from collections import defaultdict

import numpy as np
import pymatgen


# ignore (mostly) pymatgen warning about He electronegativity
warnings.filterwarnings("ignore")


# base directory in which to look for config/settings files
BASEDIR = os.path.join(os.path.dirname(__file__), "..")


try:
    from utils import classify_material_utils as cmu
except ModuleNotFoundError:
    sys.path.append(BASEDIR)
    from utils import classify_material_utils as cmu


elements_file = os.path.join(BASEDIR, "config", "ELEMENTS.json")
with open(elements_file, "r") as fr:
    ELEMENTS = json.load(fr)


PROPERTY_KEYS = [
    "formation_energy_per_atom",
    "volume_per_atom",
    "band_gap",
    "total_magnetization_per_atom",
]


def get_pot(entry_dict, elem):
    for pot in entry_dict["potentials"]:
        if pot.split("_")[0] == elem:
            return pot


def _get_stats_from_elem_data(elem_data):
    elem_stats = {}
    for db in elem_data:
        elem_stats[db] = {}
        for e in elem_data[db]:
            elem_stats[db][e] = {}
            for stat in elem_data[db][e]:
                if "pot" in stat:
                    elem_stats[db][e][stat] = dict(Counter(elem_data[db][e][stat]))
                    continue
                elem_stats[db][e].update(
                    {
                        "{}_count".format(stat): len(elem_data[db][e][stat]),
                        "{}_mean".format(stat): np.mean(elem_data[db][e][stat]),
                        "{}_median".format(stat): np.median(elem_data[db][e][stat]),
                    }
                )
    return elem_stats


def get_elem_stats_per_db(data, nz=False):
    """
    all:
        AFLOW:
            H:
                pot: H: 221, H_h: 2
                formation_count:
                formation_mean:
                formation_median:
    """
    elem_data = {}
    for db in data:
        print("  DB: {}".format(db))

        elem_data[db] = dict([(e, defaultdict(list)) for e in ELEMENTS])

        for icsd_uid in data[db]:
            assert len(data[db][icsd_uid]) == 1

            entry = data[db][icsd_uid][0]

            comp = pymatgen.Composition(entry["chemical_formula"])

            for e in comp.to_reduced_dict:
                elem_data[db][e]["pot"].append(get_pot(entry, e))

                for pk in PROPERTY_KEYS:
                    val = float(entry[pk]) if pk in entry else None
                    if val is None:
                        continue
                    if nz:
                        if "band_gap" in pk:
                            if cmu._is_metallic(val):
                                continue
                        elif "magnetization" in pk:
                            if not cmu._is_magnetic(val):
                                continue
                    elem_data[db][e][pk].append(val)

    return _get_stats_from_elem_data(elem_data)


def get_elem_stats_per_pair(data, nz=False):
    """
    all:
        AFLOW-OQMD:
            H:
                db1_pot: H: 221, H_h: 2
                db2_pot: H: 221, H_h: 2
                db1_formation_count:
                db1_formation_mean:
                db1_formation_median:
                db2_formation_count:
                db2_formation_mean:
                db2_formation_median:
    """
    elem_data = {}
    for db1, db2 in it.combinations(data.keys(), 2):
        dbc = "-".join([db1, db2])
        print("  DBC: {}".format(dbc))

        db1_uids = data[db1].keys()
        db2_uids = data[db2].keys()
        shared_uids = set(db1_uids).intersection(db2_uids)

        elem_data[dbc] = dict([(e, defaultdict(list)) for e in ELEMENTS])

        for icsd_uid in shared_uids:
            assert len(data[db1][icsd_uid]) == 1
            assert len(data[db2][icsd_uid]) == 1

            entry1 = data[db1][icsd_uid][0]
            entry2 = data[db2][icsd_uid][0]

            comp = pymatgen.Composition(entry1["chemical_formula"])

            for e in comp.to_reduced_dict:
                elem_data[dbc][e]["db1_pot"].append(get_pot(entry1, e))
                elem_data[dbc][e]["db2_pot"].append(get_pot(entry2, e))

                for pk in PROPERTY_KEYS:
                    val1 = float(entry1[pk]) if pk in entry1 else None
                    val2 = float(entry2[pk]) if pk in entry2 else None
                    if val1 is None or val2 is None:
                        continue
                    if nz:
                        if "band_gap" in pk:
                            if cmu._is_metallic(val1) or cmu._is_metallic(val2):
                                continue
                        elif "magnetization" in pk:
                            if not cmu._is_magnetic(val1) or not cmu._is_magnetic(val2):
                                continue
                    elem_data[dbc][e]["db1_{}".format(pk)].append(val1)
                    elem_data[dbc][e]["db2_{}".format(pk)].append(val2)

    return _get_stats_from_elem_data(elem_data)


if __name__ == "__main__":
    curated_data_gz = os.path.join("data", "DFTDB_ICSD_UID_curated_data.json.gz")
    with gzip.open(curated_data_gz, "rb") as fr:
        curated_data = json.load(fr)

    stat_types = ["all", "n.z."]

    # plain per-db numbers
    print("[PER DB STATS]\n")
    elem_stats_per_db = {}
    for stat_type in stat_types:
        nz = stat_type == "n.z."
        begin = time.time()
        print("Stats for n.z. only? {}".format(nz))
        elem_stats_per_db[stat_type] = get_elem_stats_per_db(curated_data, nz=nz)
        end = time.time()
        print("Done (time: {:.1f}m)".format((end - begin) / 60.0))

    elem_stats_per_db_file = os.path.join("data", "elem_stats_per_db.json")
    with open(elem_stats_per_db_file, "w") as fw:
        json.dump(elem_stats_per_db, fw, indent=2)

    # numbers for pairwise comparison of dbs
    print("\n[PAIRWISE DBC STATS]\n")
    elem_stats_per_pair = {}
    for stat_type in stat_types:
        nz = stat_type == "n.z."
        begin = time.time()
        print("Stats for n.z. only? {}".format(nz))
        elem_stats_per_pair[stat_type] = get_elem_stats_per_pair(curated_data, nz=nz)
        end = time.time()
        print("Done (time: {:.1f}m)".format((end - begin) / 60.0))

    elem_stats_per_pair_file = os.path.join("data", "elem_stats_per_pair.json")
    with open(elem_stats_per_pair_file, "w") as fw:
        json.dump(elem_stats_per_pair, fw, indent=2)
