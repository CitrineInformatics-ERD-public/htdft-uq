import os
import sys
import json
import gzip
import itertools as it
import warnings
import random
from collections import defaultdict

import numpy as np
import scipy.stats


# ignore (mostly) pymatgen warning about He electronegativity
warnings.filterwarnings("ignore")


# base directory in which to look for config/settings files
BASEDIR = os.path.dirname(__file__)


try:
    from utils import classify_material_utils
except ModuleNotFoundError:
    sys.path.append(BASEDIR)
    from utils import classify_material_utils


mat_cls_file = os.path.join(BASEDIR, "config", "MATERIAL_CLASSES.json")
with open(mat_cls_file, "r") as fr:
    MATERIAL_CLASSES = json.load(fr)


PROPERTY_KEYS = [
    "formation_energy_per_atom",
    "volume_per_atom",
    "band_gap",
    "total_magnetization_per_atom",
]


STATS = [
    "tally",
    "mean",
    "median",
    "mae",
    "mse",
    "rmse",
    "mad",
    "mpad",
    "pearson",
    "spearman",
    "kendalltau",
    "Q1",
    "Q3",
    "IQR",
]


def rprint(uid1, uid2, msg, *args, **kwargs):
    if uid1 == uid2:
        print(msg, *args, **kwargs)


def _get_tally(y1, y2):
    assert len(y1) == len(y2)
    return len(y1)


def _get_mean(y1, y2):
    return [np.mean(y1), np.mean(y2)]


def _get_median(y1, y2):
    return [np.median(y1), np.median(y2)]


def _get_mae(y1, y2):
    return np.mean(np.abs(y1 - y2))


def _get_mse(y1, y2):
    return np.mean((y1 - y2) ** 2)


def _get_rmse(y1, y2):
    return np.sqrt(_get_mse(y1, y2))


def _get_mad(y1, y2):
    return np.median(np.abs(y1 - y2))


def _get_mpad(y1, y2):
    y_mean = 0.5 * np.abs(y1 + y2)
    pct_abs_dev = np.abs(y1 - y2) * 100 / y_mean
    pct_abs_dev[y_mean < 1e-3] = 0
    return np.median(pct_abs_dev)


def _get_pearson(y1, y2):
    return scipy.stats.pearsonr(y1, y2)[0]


def _get_spearman(y1, y2):
    return scipy.stats.spearmanr(y1, y2)[0]


def _get_kendalltau(y1, y2):
    return scipy.stats.kendalltau(y1, y2)[0]


def _get_quantile(y1, y2, q):
    dy = y1 - y2
    return np.quantile(dy, q)


def _get_Q1(y1, y2):
    return _get_quantile(y1, y2, 0.25)


def _get_Q3(y1, y2):
    return _get_quantile(y1, y2, 0.75)


def _get_IQR(y1, y2):
    return _get_Q3(y1, y2) - _get_Q1(y1, y2)


def get_stats(curated_data):
    pmc_data = {}
    disagree_stats = {}
    for db1, db2 in it.combinations(curated_data.keys(), 2):
        db_combo = "-".join([db1, db2])
        print("[{}]\n".format(db_combo))

        db1_uids = curated_data[db1].keys()
        db2_uids = curated_data[db2].keys()
        shared_uids = set(db1_uids).intersection(db2_uids)

        pmc_data[db_combo] = {}
        disagree_stats[db_combo] = defaultdict(int)
        for pk in PROPERTY_KEYS:
            print("PROPERTY: {} <spot check!>".format(pk))
            pmc_data[db_combo][pk] = defaultdict(list)
            rnd_uid = random.choice(list(shared_uids))
            for uid in shared_uids:
                props1 = curated_data[db1][uid][0]
                props2 = curated_data[db2][uid][0]
                rprint(uid, rnd_uid, "Record #1:")
                rprint(uid, rnd_uid, json.dumps(props1, indent=2))
                rprint(uid, rnd_uid, "Record #2:")
                rprint(uid, rnd_uid, json.dumps(props2, indent=2))
                val1 = props1.get(pk)
                val2 = props2.get(pk)
                if val1 is None or val2 is None:
                    msg = "Property missing from at least one DB. Skipping.\n"
                    rprint(uid, rnd_uid, msg)
                    continue
                val1 = float(val1)
                val2 = float(val2)
                # get disagreement on metallic and magnetic numbers
                if pk == "band_gap":
                    if classify_material_utils._is_metallic(
                        val1
                    ) ^ classify_material_utils._is_metallic(val2):
                        disagree_stats[db_combo][pk] += 1
                elif pk == "total_magnetization_per_atom":
                    if classify_material_utils._is_magnetic(
                        val1
                    ) ^ classify_material_utils._is_magnetic(val2):
                        disagree_stats[db_combo][pk] += 1
                # skip:
                # a. band gap comparison if material is metallic in one of the
                # two databases, OR
                # b. magnetization comparison if material is nonmagnetic in one
                # of the two databases.
                if pk == "band_gap":
                    if any(
                        [classify_material_utils._is_metallic(v) for v in [val1, val2]]
                    ):
                        msg = "0 {}. Skipping.\n".format(pk)
                        rprint(uid, rnd_uid, msg)
                        continue
                elif pk == "total_magnetization_per_atom":
                    if any(
                        [
                            not classify_material_utils._is_magnetic(v)
                            for v in [val1, val2]
                        ]
                    ):
                        msg = "0 {}. Skipping.\n".format(pk)
                        rprint(uid, rnd_uid, msg)
                        continue
                # "store" the values of interest, the difference between them,
                # and some other information about the material/entry in the
                # two DBs being compared
                pmc_record = (
                    val1,
                    val2,
                    val2 - val1,
                    *[props1.get(_pk) for _pk in PROPERTY_KEYS],
                    *[props2.get(_pk) for _pk in PROPERTY_KEYS],
                    props1.get("pretty_formula"),
                    props1.get("number_of_atoms"),
                    props1.get("icsd_uid"),
                )
                pmc_data[db_combo][pk]["all"].append(pmc_record)
                for mat_cls in MATERIAL_CLASSES:
                    fn_name = "is_{}".format(mat_cls.replace("-", "_"))
                    classify_fn = getattr(classify_material_utils, fn_name)
                    try:
                        in_mat_cls = classify_fn(props1, props2)
                    except classify_material_utils.KeyNotCommonError:
                        in_mat_cls = False
                    rprint(uid, rnd_uid, "{}: {}".format(mat_cls, in_mat_cls))
                    if not in_mat_cls:
                        continue
                    pmc_data[db_combo][pk][mat_cls].append(pmc_record)
                rprint(uid, rnd_uid, "")

    # order all records in the dataset by the absolute differences
    for db_combo in pmc_data:
        for pk in pmc_data[db_combo]:
            for mat_cls in pmc_data[db_combo][pk]:
                pmc_data[db_combo][pk].update(
                    {
                        mat_cls: sorted(
                            pmc_data[db_combo][pk][mat_cls], key=lambda x: abs(x[2])
                        )
                    }
                )

    pmc_stats = {}
    for db_combo in pmc_data:
        pmc_stats[db_combo] = {}
        for pk in pmc_data[db_combo]:
            pmc_stats[db_combo][pk] = {}
            for mat_cls, values in pmc_data[db_combo][pk].items():
                y1 = np.array([float(v[0]) for v in values])
                y2 = np.array([float(v[1]) for v in values])
                pmc_stats[db_combo][pk][mat_cls] = dict(
                    [
                        (stat, globals()["_get_{}".format(stat)](y1, y2))
                        for stat in STATS
                    ]
                )
    return pmc_data, pmc_stats, disagree_stats


if __name__ == "__main__":
    curated_data_gz = os.path.join("data", "DFTDB_ICSD_UID_curated_data.json.gz")
    with gzip.open(curated_data_gz, "rb") as fr:
        curated_data = json.load(fr)

    pmc_data, pmc_stats, disagree_stats = get_stats(curated_data)

    # NB: the datafile is quite large (~1 GB)
    pmc_data_file = os.path.join("data", "pmc_data.json")
    with open(pmc_data_file, "w") as fw:
        json.dump(pmc_data, fw, indent=2)

    pmc_stats_file = os.path.join("data", "pmc_stats.json")
    with open(pmc_stats_file, "w") as fw:
        json.dump(pmc_stats, fw, indent=2)

    disagree_stats_file = os.path.join("data", "disagree_stats.json")
    with open(disagree_stats_file, "w") as fw:
        json.dump(disagree_stats, fw, indent=2)
