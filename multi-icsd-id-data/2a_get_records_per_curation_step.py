import os
import json
import gzip
import warnings
import itertools as it
from collections import defaultdict

import pymatgen

warnings.filterwarnings("ignore")


def _write_table_data(filename, data, gz=False):
    if gz:
        with gzip.open("{}.gz".format(filename), "wt") as fw:
            fw.write(json.dumps(data, indent=2))
    else:
        with open(filename, "w") as fw:
            json.dump(data, fw, indent=2)


def _print_table_row(row):
    for k, v in row.items():
        print(v, end=" ")
    print()


def _get_mp_style_reduced_formula(comp_str):
    c = pymatgen.Composition(comp_str)
    return c.get_reduced_formula_and_factor(iupac_ordering=True)[0]


def comp_mm_within_db(raw_data, ignore_uids_set, ignore_uids=True):
    """fields: db, ICSD UID, compositions"""
    table = []
    for db in raw_data:
        for icsd_uid, entries in raw_data[db].items():
            if ignore_uids and icsd_uid in ignore_uids_set[db]:
                continue
            if not len(entries) > 1:
                continue
            formulas = set([e["pretty_formula"] for e in entries])
            if len(formulas) > 1:
                row = {"db": db, "icsd_uid": icsd_uid, "compositions": list(formulas)}
                table.append(row)
                _print_table_row(row)
    return table


def comp_mm_across_dbs(raw_data, ignore_uids_set, ignore_uids=True):
    """fields: db-combination, ICSD UID, compositions"""
    table = []
    for db1, db2 in it.combinations(raw_data.keys(), 2):
        db_combo = "-".join([db1, db2])
        db1_uids = raw_data[db1].keys()
        db2_uids = raw_data[db2].keys()
        shared_uids = set(db1_uids).intersection(db2_uids)

        for uid in shared_uids:
            if ignore_uids:
                if uid in ignore_uids_set[db1] or uid in ignore_uids_set[db2]:
                    continue
            forms1 = [e["pretty_formula"] for e in raw_data[db1][uid]]
            forms2 = [e["pretty_formula"] for e in raw_data[db2][uid]]
            _forms = forms1 + forms2
            formulas = set(list(map(_get_mp_style_reduced_formula, _forms)))
            if len(formulas) > 1:
                row = {
                    "db-combination": db_combo,
                    "icsd_uid": uid,
                    "compositions": [
                        _get_mp_style_reduced_formula(list(set(forms1))[0]),
                        _get_mp_style_reduced_formula(list(set(forms2))[0]),
                    ],
                }
                table.append(row)
                _print_table_row(row)
    return table


def _get_lowest_en(entries):
    sorted_en = sorted(entries, key=lambda x: float(x["total_energy_per_atom"]))
    return sorted_en[0]


def unphys_props(raw_data, ignore_uids_set, ignore_uids=True):
    """fields: db, ICSD UID, pretty formula, property, value"""

    filters = {"volume_per_atom": (0, 150), "formation_energy_per_atom": (-5, 5)}

    def _is_prop_unphys(record, prop_key):
        return (
            float(record[prop_key]) < filters[prop_key][0]
            or float(record[prop_key]) > filters[prop_key][1]
        )

    table = []
    for db in raw_data:
        for icsd_uid, entries in raw_data[db].items():
            if ignore_uids and icsd_uid in ignore_uids_set[db]:
                continue
            record = _get_lowest_en(entries)
            for prop_key in filters:
                if record.get(prop_key) is None:
                    continue
                # ignore unphysical AFLOW boride formation energies
                if db == "AFLOW" and prop_key == "formation_energy_per_atom":
                    formula = record.get("pretty_formula")
                    c = pymatgen.Composition(formula)
                    if "B" in c.to_reduced_dict:
                        ignore_uids_set["AFLOW"].add(icsd_uid)
                        continue
                if _is_prop_unphys(record, prop_key):
                    row = {
                        "db": db,
                        "icsd_uid": icsd_uid,
                        "composition": record["pretty_formula"],
                        "property": prop_key,
                        "value": float(record.get(prop_key)),
                    }
                    table.append(row)
                    _print_table_row(row)
    return table


def _add_to_ignore_uids_set(ignore_uids_set, data):
    for record in data:
        if "db" in record:
            ignore_uids_set[record["db"]].add(record["icsd_uid"])
        elif "db-combination" in record:
            db1, db2 = record["db-combination"].split("-")
            ignore_uids_set[db1].add(record["icsd_uid"])
            ignore_uids_set[db2].add(record["icsd_uid"])


def ordered_outliers(raw_data, ignore_uids_set, ignore_uids=True):
    """fields per property: db-combination, pretty formula, delta"""
    props = [
        "formation_energy_per_atom",
        "volume_per_atom",
        "band_gap",
        "total_magnetization_per_atom",
    ]

    outliers = {}
    for db1, db2 in it.combinations(raw_data.keys(), 2):
        db_combo = "-".join([db1, db2])
        outliers[db_combo] = defaultdict(list)

        db1_uids = raw_data[db1].keys()
        db2_uids = raw_data[db2].keys()
        shared_uids = set(db1_uids).intersection(db2_uids)

        for uid in shared_uids:
            if ignore_uids:
                if uid in ignore_uids_set[db1] or uid in ignore_uids_set[db2]:
                    continue
            record1 = _get_lowest_en(raw_data[db1][uid])
            record2 = _get_lowest_en(raw_data[db2][uid])
            for prop in props:
                if prop not in record1 or prop not in record2:
                    continue
                row = (
                    uid,
                    record1["pretty_formula"],
                    float(record1[prop]),
                    float(record2[prop]),
                    float(record1[prop]) - float(record2[prop]),
                )
                outliers[db_combo][prop].append(row)
        for prop in props:
            outliers[db_combo][prop] = sorted(
                outliers[db_combo][prop], key=lambda x: -1 * abs(x[-1])
            )
    return outliers


def normalize_magmoms(enp_icsd_uid):
    for db in enp_icsd_uid:
        for icsd_uid, entries in enp_icsd_uid[db].items():
            magmom = entries[0].get("total_magnetization_per_atom")
            if magmom is None:
                continue
            formula = entries[0].get("chemical_formula")
            cdict = pymatgen.Composition(formula).to_reduced_dict
            natoms_pfu = sum(cdict.values())
            # if db == 'AFLOW' and 'Gd' in cdict and magmom > 0.1:
            #     vprint(0, icsd_uid, entries[0], formula, cdict, natoms_pfu)
            entries[0].update(
                {"total_magnetization_per_atom": abs(float(magmom)) * natoms_pfu}
            )


if __name__ == "__main__":
    raw_dat_gz = os.path.join("data", "DFTDB_ICSD_UID_raw_inv_data.json.gz")

    with gzip.open(raw_dat_gz, "rb") as fr:
        raw_data = json.load(fr)
    normalize_magmoms(raw_data)

    ignore_uids_set = defaultdict(set)

    print("Composition mismatch within DBs:")
    data = comp_mm_within_db(raw_data, ignore_uids_set, ignore_uids=True)
    _add_to_ignore_uids_set(ignore_uids_set, data)
    _write_table_data(os.path.join("data", "comp_mm_within_db.json"), data)
    print()

    print("Composition mismatch across DBs:")
    data = comp_mm_across_dbs(raw_data, ignore_uids_set, ignore_uids=True)
    _add_to_ignore_uids_set(ignore_uids_set, data)
    _write_table_data(os.path.join("data", "comp_mm_across_dbs.json"), data)
    print()

    print("Unphysical properties:")
    data = unphys_props(raw_data, ignore_uids_set, ignore_uids=True)
    _add_to_ignore_uids_set(ignore_uids_set, data)
    _write_table_data(os.path.join("data", "unphys_props.json"), data)
    print()

    print("Getting an ordered list of outliers...", end=" ")
    data = ordered_outliers(raw_data, ignore_uids_set, ignore_uids=True)
    _write_table_data(os.path.join("data", "ordered_outliers.json"), data, gz=True)
    print("done.")
