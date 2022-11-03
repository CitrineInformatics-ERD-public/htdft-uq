import os
import json
import gzip
import itertools as it
import warnings
from collections import defaultdict
import random
import time

import pymatgen

# do not spoil the pristine stdout space LUL
warnings.filterwarnings("ignore")


# global standard output verbosity
# everything *UP TO* the level is printed to standard output
VERBOSITY_LEVEL = 2


# root directory in which to look for config/settings JSON files
BASEDIR = os.path.dirname(__file__)


# Module level list `DATASETS` with the names of the three databases
datasets_file = os.path.join(BASEDIR, "config", "DATASETS.json")
with open(datasets_file, "r") as fr:
    DATASETS = json.load(fr)


def vprint(vlevel, *args, **kwargs):
    """Print iff `vlevel` is within global `VERBOSITY_LEVEL`."""
    if vlevel <= VERBOSITY_LEVEL:
        print(*args, **kwargs)


def _frozenset_to_str(f):
    return "-".join(sorted(f))


def _jsonify_frozenset_dict(data):
    if isinstance(data, frozenset):
        return _frozenset_to_str(data)
    elif isinstance(data, list):
        return [_jsonify_frozenset_dict(d) for d in data]
    elif isinstance(data, dict):
        json_dict = {}
        for k, v in data.items():
            json_dict.update({_jsonify_frozenset_dict(k): _jsonify_frozenset_dict(v)})
        return json_dict
    else:
        return data


def _json_dumps(data):
    return json.dumps(_jsonify_frozenset_dict(data), indent=2)


def _print_n_entries(ddict, header="# problem entries:"):
    vprint(1, "{} {}".format(header, dict([(k, len(v)) for k, v in ddict.items()])))


def _print_pprop_tally(ddict):
    pprops = {
        "formation_energy_per_atom": "Formation energy",
        "volume_per_atom": "Volume/atom",
        "band_gap": "Band gap",
        "total_magnetization_per_atom": "Magnetization/atom",
    }
    pprop_tally = {}
    vprint(1, "Per-property tally of records:")
    dbc = {}
    header = ["{:<20s}".format("property")]
    for n in [1, 2, 3]:
        dbc[n] = [
            "-".join([db for db in _dbc])
            for _dbc in sorted(list(it.combinations(ddict.keys(), n)))
        ]
        fformat = "{{:^{}s}}".format(max([len(db) for db in dbc[n]]) + 1)
        header.extend([fformat.format(db) for db in dbc[n]])
    vprint(1, " ".join(header))

    def _db_keys(db, prop):
        return set([k for k, v in ddict[db].items() if prop in v[0]])

    for prop in pprops:
        row = ["{:<20s}".format(pprops[prop])]
        pprop_tally[prop] = {}
        for n, dbs in dbc.items():
            fformat = "{{:^{}s}}".format(max([len(db) for db in dbs]) + 1)
            for db in dbs:
                try:
                    count = len([k for k, v in ddict[db].items() if prop in v[0]])
                except KeyError:
                    sp_db = db.split("-")
                    common_keys = _db_keys(sp_db[0], prop).intersection(
                        *[_db_keys(_db, prop) for _db in sp_db]
                    )
                    count = len(common_keys)
                pprop_tally[prop].update({db: count})
                row.append(fformat.format(str(count)))
        vprint(1, " ".join(row))
    vprint(1)
    return pprop_tally


def _print_tally(ddict, header="# per-DB entries/ # pair-wise common entries"):
    vprint(1, header)
    dbs = sorted(list(ddict.keys()))
    db_combos = sorted(list(it.combinations(dbs, 2)))
    for db, db_combo in zip(dbs, db_combos):
        db1, db2 = db_combo
        common_keys = set(ddict[db1].keys()).intersection(ddict[db2].keys())
        db_combo_str = "-".join([db1, db2])
        vprint(
            1,
            "{:>6s}: {:<6d} | {:>12s}: {:<6d}".format(
                db, len(ddict[db]), db_combo_str, len(common_keys)
            ),
        )
    vprint(1)


def _get_icsd_ids(prop_dict):
    icsd_dicts = filter(lambda x: x["name"] == "ICSD", prop_dict["ids"])
    icsd_ids = sorted(map(lambda x: x["value"], icsd_dicts))
    return icsd_ids


def _get_mp_style_reduced_formula(comp_str):
    c = pymatgen.Composition(comp_str)
    return c.get_reduced_formula_and_factor(iupac_ordering=True)[0]


def _spot_check_annotation(xtr_props, mpm_props, non_mpm_props):
    vprint(2, "<Spot Check!>")

    vprint(2, "Sample annotated MP entry:")
    vprint(2, _json_dumps(random.choice(xtr_props["MP"])))
    vprint(2)

    vprint(2, "Sample AFLOW entry *without* ICSD ID match in MP:")
    vprint(2, _json_dumps(random.choice(non_mpm_props["AFLOW"])))
    vprint(2)

    vprint(2, "Sample OQMD entry *with* ICSD ID match in MP:")
    vprint(2, _json_dumps(random.choice(mpm_props["OQMD"])))
    vprint(2)


def _get_mp_icsd_uids(extracted_props):
    # generate a set of ICSD UIDs from MP
    # MUST be done *before* annotating MP records!
    icsd_uids = set()
    for this_prop in extracted_props["MP"]:
        icsd_ids = _get_icsd_ids(this_prop)
        this_uid = frozenset(icsd_ids)
        uid_matched = False
        matching_uids = set()
        for uid in icsd_uids:
            if this_uid.intersection(uid):
                uid_matched = True
                matching_uids.add(uid)
        if not uid_matched:
            icsd_uids.add(this_uid)
        else:
            for uid in matching_uids:
                icsd_uids.remove(uid)
            icsd_uids.add(this_uid.union(*matching_uids))
    return icsd_uids


def annotate_with_icsd_uids(extracted_props):
    """
    Annotates each entry in the `extracted_props` dictionary with a new key,
    `icsd_uid`, a frozenset of all ICSD IDs corresponding to the same entry (as
    categorized by Materials Project). For ICSD IDs in AFLOW or OQMD but not in
    Materials Project, the `icsd_uid` is simply a n=1 frozenset of the ICSD ID.
    """
    vprint(0, "[*] ANNOTATE WITH ICSD UID")

    vprint(1, "Getting ICSD UIDs from MP...")
    icsd_uids = _get_mp_icsd_uids(extracted_props)
    vprint(1, "# ICSD UIDs from MP: {}".format(len(icsd_uids)))

    # annotate MP records
    vprint(1, "Annotating MP records with ICSD UID...")
    mp_inv_uid = {}
    for this_prop in extracted_props["MP"]:
        icsd_ids = _get_icsd_ids(this_prop)
        this_uid = frozenset(icsd_ids)
        # uid_match = [uid for uid in icsd_uids if this_uid.issubset(uid)]
        uid_match = list(filter(lambda x: this_uid.issubset(x), icsd_uids))
        try:
            assert len(uid_match) == 1
        except AssertionError:
            print("Unique UID error:")
            print(
                "Property, UID, UID matches: {}, {}, {}".format(
                    this_prop, this_uid, uid_match
                )
            )
            raise
        this_prop.update({"icsd_uid": uid_match[0]})
        for icsd_id in icsd_ids:
            mp_inv_uid[icsd_id] = this_prop["icsd_uid"]
    vprint(1, "# ICSD IDs in MP: {}".format(len(mp_inv_uid)))

    # the following two dictionaries are only used for spot-checks
    mpm_props = defaultdict(list)
    non_mpm_props = defaultdict(list)
    vprint(1, "Annotating AFLOW and OQMD records with ICSD UID...")
    for db in ["AFLOW", "OQMD"]:
        for this_prop in extracted_props[db]:
            icsd_ids = _get_icsd_ids(this_prop)
            assert len(icsd_ids) == 1
            icsd_uid = mp_inv_uid.get(icsd_ids[0], frozenset(icsd_ids))
            this_prop.update({"icsd_uid": icsd_uid})
            if icsd_uid == frozenset(icsd_ids):
                non_mpm_props[db].append(this_prop)
            else:
                mpm_props[db].append(this_prop)
        vprint(1, "# non-MP ICSD IDs in {}: {}".format(db, len(non_mpm_props[db])))

    vprint(1, "Extracted properties have been annotated with ICSD UID.\n")
    _spot_check_annotation(extracted_props, mpm_props, non_mpm_props)


def _spot_check_dict_inversion(xtr_props, enp_icsd_uid):
    vprint(2, "<Spot Check!>")

    db = random.choice(DATASETS)
    random_prop = random.choice([p for p in xtr_props[db] if len(p["icsd_uid"]) > 1])
    vprint(2, "Sample pre-inversion extracted properties dict:")
    vprint(2, _json_dumps(random_prop))
    vprint(2)
    vprint(2, "Corresponding post-inversion entries per ICSD UID dict:")
    vprint(2, _json_dumps(enp_icsd_uid[db][random_prop["icsd_uid"]]))
    vprint(2)


def get_entries_per_icsd_uid(extracted_props):
    """
    Inverts the `extracted_props` dictionary so that keys are unique `icsd_uid`
    frozensets and values are a *list* of extracted properties of all entries
    corresponding to the `icsd_uid`, and returns the inverted dict.
    """
    vprint(0, "[*] INVERT EXTRACT PROPERTIES DICT")

    enp_icsd_uid = {}
    dup_count = defaultdict(int)
    for db in DATASETS:
        enp_icsd_uid[db] = {}
        for p_dict in extracted_props[db]:
            if p_dict["icsd_uid"] in enp_icsd_uid[db]:
                dup_count[db] += 1
                enp_icsd_uid[db][p_dict["icsd_uid"]].append(p_dict)
            else:
                enp_icsd_uid[db][p_dict["icsd_uid"]] = [p_dict]

    vprint(1, "# duplicate entries: {}".format(dict(dup_count)))
    vprint(
        1,
        "Extracted properties have been inverted into an"
        " entries-per-ICSD-UID dictionary.\n",
    )
    _spot_check_dict_inversion(extracted_props, enp_icsd_uid)
    _print_tally(enp_icsd_uid)
    _print_pprop_tally(enp_icsd_uid)

    return enp_icsd_uid


def rm_within_db_comp_mismatch(enp_icsd_uid):
    """
    Removes ICSD UIDs from databases if the multiple extracted property
    entries under that ICSD UID do not have matching compositions.
    """
    vprint(0, "[*] REMOVE WITHIN-DB COMPOSITION MISMATCH")

    ens_comp_mm = defaultdict(list)
    for db in DATASETS:
        for icsd_uid, entries in enp_icsd_uid[db].items():
            if not len(entries) > 1:
                continue
            formulas = set([e["pretty_formula"] for e in entries])
            if len(formulas) > 1:
                ens_comp_mm[db].append(icsd_uid)
    ens_comp_mm = dict(ens_comp_mm)

    _print_n_entries(ens_comp_mm)

    vprint(2)
    for db, icsd_uids in ens_comp_mm.items():
        vprint(2, "({}) mismatched compositions:".format(db))
        for icsd_uid in icsd_uids:
            prob_en = enp_icsd_uid[db].pop(icsd_uid)
            vprint(
                2, "  {}".format(", ".join(set([e["pretty_formula"] for e in prob_en])))
            )
        vprint(2)

    vprint(1, "Within-DB composition inconsistencies have been removed.\n")
    _print_tally(enp_icsd_uid)


def rm_across_db_comp_mismatch(enp_icsd_uid):
    """
    Removes ICSD UIDs from databases if the composition corresponding to the
    ICSD UID differs across any two databases.
    """
    vprint(0, "[*] REMOVE ACROSS-DBs COMPOSITION MISMATCH")
    ens_comp_mm = defaultdict(list)
    mm_comps = defaultdict(dict)

    for db1, db2 in it.combinations(DATASETS, 2):
        db_combo = "{}-{}".format(db1, db2)
        db1_icsd_uids = enp_icsd_uid[db1].keys()
        db2_icsd_uids = enp_icsd_uid[db2].keys()
        shared_uids = set(db1_icsd_uids).intersection(db2_icsd_uids)

        for icsd_uid in shared_uids:
            _forms = [e["pretty_formula"] for e in enp_icsd_uid[db1][icsd_uid]]
            _forms.extend([e["pretty_formula"] for e in enp_icsd_uid[db2][icsd_uid]])
            formulas = set(list(map(_get_mp_style_reduced_formula, _forms)))
            if len(formulas) > 1:
                ens_comp_mm[db_combo].append(icsd_uid)
                mm_comps[db_combo][icsd_uid] = list(formulas)
    ens_comp_mm = dict(ens_comp_mm)
    mm_comps = dict(mm_comps)

    _print_n_entries(ens_comp_mm)

    vprint(2)
    to_del_uids = defaultdict(set)
    for db_combo, icsd_uids in ens_comp_mm.items():
        db1, db2 = db_combo.split("-")
        vprint(2, "({}) mismatched compositions:".format(db_combo))
        for icsd_uid in icsd_uids:
            to_del_uids[db1].add(icsd_uid)
            to_del_uids[db2].add(icsd_uid)
            vprint(2, "  {}".format(", ".join(mm_comps[db_combo][icsd_uid])))
        vprint(2)
    for db, uids in to_del_uids.items():
        for uid in uids:
            del enp_icsd_uid[db][uid]

    vprint(1, "Composition inconsistencies have been removed across DBs.\n")
    _print_tally(enp_icsd_uid)


def _spot_check_energy_filter(enp_icsd_uid, db, uid, entries):
    vprint(2, "<Spot Check!>")

    vprint(2, "Sample record with multiple entries (pre-filter):")
    vprint(2, _json_dumps(entries))
    vprint(2)
    vprint(2, "Corresponding post-filter record:")
    vprint(2, _json_dumps(enp_icsd_uid[db][uid]))
    vprint(2)


def filter_lowest_energy_entries(enp_icsd_uid):
    """Retains only the lowest energy entry per ICSD UID."""
    vprint(0, "[*] FILTER FOR LOWEST ENERGY ENTRY")

    multi_ens = defaultdict(list)
    for db in DATASETS:
        for icsd_uid, entries in enp_icsd_uid[db].items():
            if len(entries) > 1:
                multi_ens[db].append(icsd_uid)
    # the following few variables are only used for spot-checks
    _rdb = random.choice(DATASETS)
    _ruid = random.choice(multi_ens[_rdb])
    _rentries = enp_icsd_uid[_rdb][_ruid]

    header = "# ICSD UIDs with multiple entries:"
    _print_n_entries(multi_ens, header=header)

    for db, icsd_uids in multi_ens.items():
        for icsd_uid in icsd_uids:
            assert all(
                ["total_energy_per_atom" in e for e in enp_icsd_uid[db][icsd_uid]]
            )
            sorted_en = sorted(
                enp_icsd_uid[db][icsd_uid],
                key=lambda x: float(x["total_energy_per_atom"]),
            )
            enp_icsd_uid[db].update({icsd_uid: [sorted_en[0]]})

    vprint(1, "There is now only one entry (lowest energy) per ICSD UID.\n")
    _spot_check_energy_filter(enp_icsd_uid, _rdb, _ruid, _rentries)
    _print_tally(enp_icsd_uid)


def _rm_unphys_prop(prop_key, enp_icsd_uid, max_val=10.0, min_val=0.0, prop_units="eV"):
    unphys_ids = defaultdict(list)
    for db in DATASETS:
        for icsd_uid, entries in enp_icsd_uid[db].items():
            prop_val = entries[0].get(prop_key, None)
            if prop_val is None:
                continue
            if float(prop_val) > max_val:
                unphys_ids[db].append(icsd_uid)
            elif float(prop_val) < min_val:
                unphys_ids[db].append(icsd_uid)

    header = f"# entries with `{prop_key}` outside ({min_val}, {max_val}) {prop_units}:"
    _print_n_entries(unphys_ids, header=header)

    vprint(2)
    for db, icsd_uids in unphys_ids.items():
        vprint(2, "({}) unphysical {}:".format(db, prop_key))
        for icsd_uid in icsd_uids:
            vprint(
                2,
                "  {:>10s} {:.2f} {}".format(
                    enp_icsd_uid[db][icsd_uid][0]["pretty_formula"],
                    float(enp_icsd_uid[db][icsd_uid][0][prop_key]),
                    list(icsd_uid),
                ),
            )
            del enp_icsd_uid[db][icsd_uid]
        vprint(2)

    vprint(1, "Unphysical {} have been removed.\n".format(prop_key))
    _print_tally(enp_icsd_uid)


def rm_unphysical_formation_energies(enp_icsd_uid, max_form=5.0, min_form=-5.0):
    """
    Removes ICSD UIDs from databases if the corresponding per-atom formation
    energy value is unphysical (outside the specified thresholds).
    """
    vprint(0, "[*] REMOVE UNPHYSICAL FORMATION ENERGIES")

    _rm_unphys_prop(
        "formation_energy_per_atom",
        enp_icsd_uid,
        max_val=max_form,
        min_val=min_form,
        prop_units="eV/atom",
    )


def rm_unphysical_volumes(enp_icsd_uid, max_vol=150.0):
    """
    Removes ICSD UIDs from databases if the corresponding per-atom volume
    value is unphysical (greater than the specified threshold).
    """
    vprint(0, "[*] REMOVE UNPHYSICAL VOLUMES")

    _rm_unphys_prop(
        "volume_per_atom", enp_icsd_uid, max_val=max_vol, prop_units="Ang^3/atom"
    )


def rm_boride_formation_energies_from_aflow(enp_icsd_uid):
    """Removes formation energies from B-containing materials in AFLOW."""
    vprint(0, "[*] REMOVE AFLOW BORIDE FORMATION ENERGIES")

    vprint(2)
    for icsd_uid, entries in enp_icsd_uid["AFLOW"].items():
        c = pymatgen.Composition(entries[0].get("pretty_formula"))
        f = entries[0].get("formation_energy_per_atom", None)
        if f is not None and "B" in c.to_reduced_dict:
            vprint(2, "  {:>14s}: {:.3f}".format(c.reduced_formula, float(f)))
            del entries[0]["formation_energy_per_atom"]
    vprint(2)

    vprint(1, "AFLOW boron formation energies have been removed.\n")
    _print_tally(enp_icsd_uid)


def absolutize_magmoms(enp_icsd_uid):
    """Converts queried per-atom total magnetization to absolute values."""
    vprint(0, "[*] CONVERT TO ABSOLUTE MAGNETIZATION")

    for db in DATASETS:
        for icsd_uid, entries in enp_icsd_uid[db].items():
            magmom = entries[0].get("total_magnetization_per_atom")
            if magmom is None:
                continue
            entries[0].update({"total_magnetization_per_atom": abs(float(magmom))})

    vprint(1, "All per-atom total magnetization are now absolute values.\n")


def normalize_magmoms(enp_icsd_uid):
    for db in DATASETS:
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


def get_single_icsd_uids(inverted_props_dict):
    """Removes all keys with multiple ICSD IDs from the dict."""
    vprint(0, "[*] FILTER FOR UID'S WITH ONE ICSD ID")

    single_ddict = {}
    for db in inverted_props_dict:
        single_ddict[db] = {}
        for icsd_uid, entries in inverted_props_dict[db].items():
            if len(icsd_uid) != 1:
                continue
            single_ddict[db][icsd_uid] = entries
    vprint(1, "UIDs composed of multiple ICSD IDs have been removed.\n")
    _print_tally(single_ddict)
    _print_pprop_tally(single_ddict)
    return single_ddict


def _write_ddict_to_disk(fname, ddict):
    _ddict = _jsonify_frozenset_dict(ddict)
    with gzip.open(fname, "wt") as fw:
        fw.write(json.dumps(_ddict, indent=2))


def curate_data(extracted_props, write_inv_dict=True, single_icsd_uid=True):
    """Runs the curation pipeline and returns the curated data dictionary."""
    vprint(1, "\n[DATA CURATION]\n")

    # annotate every entry in each of the three databases with the
    # corresponding "ICSD UID", a hyphen-separated list of ICSD IDs
    # corresponding to the entry from MP
    annotate_with_icsd_uids(extracted_props)

    # invert the dataset so that keys are unique ICSD IDs and values are all
    # the entries matched to the corresponding UIDs
    inverted_props_dict = get_entries_per_icsd_uid(extracted_props)

    # filter for entries that have a single ICSD ID associated with them (to
    # avoid depending on MP structure matching as the truth).
    if single_icsd_uid:
        entries_per_icsd_uid = get_single_icsd_uids(inverted_props_dict)
    else:
        entries_per_icsd_uid = inverted_props_dict

    # write the inverted dictionary to disk (for other analysis)
    if write_inv_dict:
        raw_data_gz = os.path.join("data", "DFTDB_ICSD_UID_raw_inv_data.json.gz")
        _write_ddict_to_disk(raw_data_gz, entries_per_icsd_uid)

    # do some data curation:
    # 1. remove composition mismatches (within and across databases)
    # 2. in case of multiple entries per UID, select the lowest-energy entry
    # 3. remove boride formation energies from AFLOW
    # 4. remove unphysical values of formation energy from all three DBs
    # 5. remove unphysical values of volume from all three DBs
    # 6. convert total magnetization to absolute values and normalize per
    #    formula unit (instead of per atom)
    rm_within_db_comp_mismatch(entries_per_icsd_uid)
    rm_across_db_comp_mismatch(entries_per_icsd_uid)
    filter_lowest_energy_entries(entries_per_icsd_uid)
    rm_boride_formation_energies_from_aflow(entries_per_icsd_uid)
    rm_unphysical_formation_energies(entries_per_icsd_uid)
    rm_unphysical_volumes(entries_per_icsd_uid)
    absolutize_magmoms(entries_per_icsd_uid)
    normalize_magmoms(entries_per_icsd_uid)

    vprint(1, "Data curation is complete! Final tally:\n")
    _print_tally(entries_per_icsd_uid)
    pprop_tally = _print_pprop_tally(entries_per_icsd_uid)
    pprop_tally_file = os.path.join("data", "post-curation_pprop_tally.json")
    with open(pprop_tally_file, "w") as fw:
        json.dump(pprop_tally, fw, indent=2)

    return entries_per_icsd_uid


if __name__ == "__main__":
    start_time = time.time()

    extracted_props_gz = os.path.join(BASEDIR, "data", "DFTDB_extracted_props.json.gz")
    vprint(
        0, 'Reading extracted materials properties from "{}"'.format(extracted_props_gz)
    )
    with gzip.open(extracted_props_gz, "rb") as fr:
        extracted_props = json.load(fr)

    header = "Number of loaded materials properties:"
    _print_n_entries(extracted_props, header=header)

    curated_data = curate_data(extracted_props)

    curated_data_gz = os.path.join("data", "DFTDB_ICSD_UID_curated_data.json.gz")
    _write_ddict_to_disk(curated_data_gz, curated_data)
    vprint(0, 'Curated data written to file "{}".'.format(curated_data_gz))

    end_time = time.time()
    tot_s = end_time - start_time
    print("TOTAL RUN TIME: {}m {:.1f}s".format(int(tot_s / 60), tot_s % 60))
