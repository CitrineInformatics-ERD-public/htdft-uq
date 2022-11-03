import os
import json

from pymatgen import Composition


# base directory to read config/settings from
BASEDIR = os.path.dirname(__file__)


classify_criteria_file = os.path.join(BASEDIR, "..", "config", "CLASSIFY_CRITERIA.json")
with open(classify_criteria_file, "r") as fr:
    CLASSIFY_CRITERIA = json.load(fr)


ELEMENT_GROUPS = {
    "pnictogen": ["N", "P", "As", "Sb", "Bi"],
    "chalcogen": ["S", "Se", "Te"],
    "halogen": ["F", "Cl", "Br", "I"],
    "alkali": ["Li", "Na", "K", "Rb", "Cs"],
    "alkaline_earth": ["Be", "Mg", "Ca", "Sr", "Ba"],
    "transition_metal": [
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "La",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
    ],
    "metalloid": ["B", "Ge", "Si", "Sb", "Te", "As"],
    "rare_earth": [
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ],
    "actinide": ["Ac", "Th", "Pa", "U", "Np", "Pu"],
    "noble_gas": ["He", "Ne", "Ar", "Kr", "Xe"],
    "nonmetal": ["H", "C", "N", "P", "O", "S", "Se", "F", "Cl", "Br", "I"],
}

ELEMENT_GROUPS.update(
    {
        "metal": ["Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi"]
        + ELEMENT_GROUPS["alkali"]
        + ELEMENT_GROUPS["alkaline_earth"]
        + ELEMENT_GROUPS["transition_metal"]
        + ELEMENT_GROUPS["rare_earth"]
        + ELEMENT_GROUPS["actinide"],
    }
)


class KeyNotCommonError(Exception):
    pass


def is_oxide(props1, props2):
    return "O" in Composition(props1["pretty_formula"]).to_reduced_dict


def is_nitride(props1, props2):
    return "N" in Composition(props1["pretty_formula"]).to_reduced_dict


def _any_elem_in_group(props, elem_group):
    return any(
        [
            e in ELEMENT_GROUPS[elem_group]
            for e in Composition(props["pretty_formula"]).to_reduced_dict
        ]
    )


def is_pnictide(props1, props2):
    return _any_elem_in_group(props1, "pnictogen")


def is_chalcogenide(props1, props2):
    return _any_elem_in_group(props1, "chalcogen")


def is_halide(props1, props2):
    return _any_elem_in_group(props1, "halogen")


def is_alkali_metal(props1, props2):
    return _any_elem_in_group(props1, "alkali")


def is_alkaline_earth_metal(props1, props2):
    return _any_elem_in_group(props1, "alkaline_earth")


def is_transition_metal(props1, props2):
    return _any_elem_in_group(props1, "transition_metal")


def is_metalloid(props1, props2):
    return _any_elem_in_group(props1, "metalloid")


def is_rare_earth(props1, props2):
    return _any_elem_in_group(props1, "rare_earth")


def is_actinide(props1, props2):
    return _any_elem_in_group(props1, "actinide")


def is_noble_gas(props1, props2):
    return _any_elem_in_group(props1, "noble_gas")


def _is_metal(props1):
    return _any_elem_in_group(props1, "metal")


def _is_nonmetal(props1):
    return _any_elem_in_group(props1, "nonmetal")


def is_metal_nonmetal(props1, props2):
    return _is_metal(props1) and _is_nonmetal(props1)


def is_intermetallic(props1, props2):
    return all(
        [
            e in ELEMENT_GROUPS["metal"]
            for e in Composition(props1["pretty_formula"]).to_reduced_dict
        ]
    )


def _is_magnetic(magmom_pa):
    return float(magmom_pa) > CLASSIFY_CRITERIA["is_magnetic"]


def _is_non_magnetic(magmom_pa):
    return not _is_magnetic(magmom_pa)


def _check_key_common(props1, props2, key):
    val1 = props1.get(key)
    val2 = props2.get(key)
    if val1 is None or val2 is None:
        raise KeyNotCommonError


def is_magnetic(props1, props2):
    _check_key_common(props1, props2, "total_magnetization_per_atom")
    return _is_magnetic(props1["total_magnetization_per_atom"]) and _is_magnetic(
        props2["total_magnetization_per_atom"]
    )


def is_non_magnetic(props1, props2):
    _check_key_common(props1, props2, "total_magnetization_per_atom")
    return _is_non_magnetic(
        props1["total_magnetization_per_atom"]
    ) and _is_non_magnetic(props2["total_magnetization_per_atom"])


def is_disagree_on_magnetic(props1, props2):
    _check_key_common(props1, props2, "total_magnetization_per_atom")
    return _is_magnetic(props1["total_magnetization_per_atom"]) ^ _is_magnetic(
        props2["total_magnetization_per_atom"]
    )


def _is_metallic(bg):
    return float(bg) <= CLASSIFY_CRITERIA["is_metallic"]


def _is_semiconductor(bg):
    bg_low = CLASSIFY_CRITERIA["is_metallic"]
    bg_high = CLASSIFY_CRITERIA["is_semiconductor"]
    return bg_low < float(bg) <= bg_high


def _is_insulator(bg):
    bg_high = CLASSIFY_CRITERIA["is_semiconductor"]
    return float(bg) > bg_high


def is_metallic(props1, props2):
    _check_key_common(props1, props2, "band_gap")
    return _is_metallic(props1["band_gap"]) and _is_metallic(props2["band_gap"])


def is_semiconductor(props1, props2):
    _check_key_common(props1, props2, "band_gap")
    return _is_semiconductor(props1["band_gap"]) and _is_semiconductor(
        props2["band_gap"]
    )


def is_insulator(props1, props2):
    _check_key_common(props1, props2, "band_gap")
    return _is_insulator(props1["band_gap"]) and _is_insulator(props2["band_gap"])


def is_disagree_on_metallic(props1, props2):
    _check_key_common(props1, props2, "band_gap")
    return _is_metallic(props1["band_gap"]) ^ _is_metallic(props2["band_gap"])


def is_element(props1, props2):
    return len(Composition(props1["pretty_formula"]).to_reduced_dict) == 1


def is_binary(props1, props2):
    return len(Composition(props1["pretty_formula"]).to_reduced_dict) == 2


def is_ternary(props1, props2):
    return len(Composition(props1["pretty_formula"]).to_reduced_dict) == 3


def is_quaternary(props1, props2):
    return len(Composition(props1["pretty_formula"]).to_reduced_dict) == 4


def is_pseudopotentials_agree(props1, props2):
    return set(props1["potentials"]) == set(props2["potentials"])


def is_pseudopotentials_disagree(props1, props2):
    return not is_pseudopotentials_agree(props1, props2)


def is_use_ggau(props1, props2):
    _check_key_common(props1, props2, "is_hubbard")
    return props1["is_hubbard"] == "True" and props2["is_hubbard"] == "True"


def is_use_gga(props1, props2):
    _check_key_common(props1, props2, "is_hubbard")
    return props1["is_hubbard"] == "False" and props2["is_hubbard"] == "False"


def is_disagree_on_gga_ggau(props1, props2):
    _check_key_common(props1, props2, "is_hubbard")
    return set([props1["is_hubbard"], props2["is_hubbard"]]) == set(["True", "False"])


def _is_triclinic(spg):
    return int(spg) <= 2


def _is_monoclinic(spg):
    return 2 < int(spg) <= 15


def _is_orthorhombic(spg):
    return 16 < int(spg) <= 74


def _is_tetragonal(spg):
    return 75 < int(spg) <= 142


def _is_trigonal(spg):
    return 142 < int(spg) <= 167


def _is_hexagonal(spg):
    return 168 < int(spg) <= 194


def _is_cubic(spg):
    return int(spg) > 194


def is_triclinic(props1, props2):
    return all([_is_triclinic(p["space_group_number"]) for p in [props1, props2]])


def is_monoclinic(props1, props2):
    return all([_is_monoclinic(p["space_group_number"]) for p in [props1, props2]])


def is_orthorhombic(props1, props2):
    return all([_is_orthorhombic(p["space_group_number"]) for p in [props1, props2]])


def is_tetragonal(props1, props2):
    return all([_is_tetragonal(p["space_group_number"]) for p in [props1, props2]])


def is_trigonal(props1, props2):
    return all([_is_trigonal(p["space_group_number"]) for p in [props1, props2]])


def is_hexagonal(props1, props2):
    return all([_is_hexagonal(p["space_group_number"]) for p in [props1, props2]])


def is_cubic(props1, props2):
    return all([_is_cubic(p["space_group_number"]) for p in [props1, props2]])
