import pytest
from mixer.parser import ITPTopology


def test_merge_union_atoms(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    atom_names = [a["atom"] for a in merge_top_a.atoms]
    assert "BB"  in atom_names
    assert "SC1" in atom_names
    assert "SC2" in atom_names
    assert "SC3" in atom_names
    assert "SC4" in atom_names
    assert "VS1" in atom_names
    assert "VS2" in atom_names
    assert len(merge_top_a.atoms) == 7

def test_merge_remove_redundant_drops_duplicate_bonds(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    bb_sc1_count = sum(1 for b in merge_top_a.bonds
                       if set((b["i"], b["j"])) == {"BB", "SC1"})
    assert bb_sc1_count == 1

def test_merge_remove_redundant_false_keeps_duplicate_bonds(merge_top_a, merge_top_b):
    with pytest.warns(UserWarning):
        merge_top_a.merge(merge_top_b, remove_redundant=False)
    bb_sc1_count = sum(1 for b in merge_top_a.bonds
                       if set((b["i"], b["j"])) == {"BB", "SC1"})
    assert bb_sc1_count == 2

def test_merge_remove_redundant_false_keeps_duplicate_angles(merge_top_a, merge_top_b):
    with pytest.warns(UserWarning):
        merge_top_a.merge(merge_top_b, remove_redundant=False)
    bb_sc1_sc3_count = sum(1 for a in merge_top_a.angles
                           if {a["i"], a["j"], a["k"]} == {"BB", "SC1", "SC3"})
    assert bb_sc1_sc3_count == 2

def test_merge_remove_redundant_false_warns_for_atoms(merge_top_a, merge_top_b):
    with pytest.warns(UserWarning, match="remove_redundant=False has no effect on"):
        merge_top_a.merge(merge_top_b, remove_redundant=False)

def test_merge_keep_second_false_keeps_self_atom(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True, keep_second=False)
    bb = next(a for a in merge_top_a.atoms if a["atom"] == "BB")
    assert bb["type"] == "P4"

def test_merge_keep_second_true_replaces_atom(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True, keep_second=True)
    bb = next(a for a in merge_top_a.atoms if a["atom"] == "BB")
    assert bb["type"] == "P3"

def test_merge_keep_second_false_keeps_self_bond_params(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True, keep_second=False)
    bb_sc1 = next(b for b in merge_top_a.bonds
                  if set((b["i"], b["j"])) == {"BB", "SC1"})
    assert bb_sc1["params"] == ["1", "0.47", "3800"]

def test_merge_keep_second_true_replaces_bond_params(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True, keep_second=True)
    bb_sc1 = next(b for b in merge_top_a.bonds
                  if set((b["i"], b["j"])) == {"BB", "SC1"})
    assert bb_sc1["params"] == ["1", "0.35", "2000"]

def test_merge_keep_second_false_keeps_self_moleculetype(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True, keep_second=False)
    assert merge_top_a.moleculetype["name"] == "MOLA"

def test_merge_keep_second_true_replaces_moleculetype(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True, keep_second=True)
    assert merge_top_a.moleculetype["name"] == "MOLB"

def test_merge_duplicate_angles_removed(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    bb_sc1_sc3_count = sum(1 for a in merge_top_a.angles
                           if {a["i"], a["j"], a["k"]} == {"BB", "SC1", "SC3"})
    assert bb_sc1_sc3_count == 1

def test_merge_unique_angles_kept(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    angle_sets = [{a["i"], a["j"], a["k"]} for a in merge_top_a.angles]
    assert {"BB", "SC1", "SC2"} in angle_sets
    assert {"BB", "SC1", "SC4"} in angle_sets
    assert len(merge_top_a.angles) == 3

def test_merge_duplicate_virtual_sites2_removed(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    vs1_count = sum(1 for vs in merge_top_a.virtual_sites2
                    if vs["i"] == "VS1")
    assert vs1_count == 1

def test_merge_unique_virtual_sites2_kept(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    vs_atoms = [vs["i"] for vs in merge_top_a.virtual_sites2]
    assert "VS1" in vs_atoms
    assert "VS2" in vs_atoms
    assert len(merge_top_a.virtual_sites2) == 2

def test_merge_duplicate_virtual_sitesn_removed(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    vs1_count = sum(1 for vs in merge_top_a.virtual_sitesn
                    if vs["vs"] == "VS1")
    assert vs1_count == 1

def test_merge_duplicate_exclusions_removed(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    bb_sc1_count = sum(1 for e in merge_top_a.exclusions
                       if set(e) == {"BB", "SC1"})
    assert bb_sc1_count == 1

def test_merge_unique_exclusions_kept(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    excl_sets = [set(e) for e in merge_top_a.exclusions]
    assert {"BB", "SC1", "SC2"} in excl_sets
    assert {"BB", "SC1", "SC3"} in excl_sets
    assert len(merge_top_a.exclusions) == 3

def test_merge_unique_bonds_kept(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    bond_pairs = [set((b["i"], b["j"])) for b in merge_top_a.bonds]
    assert {"SC1", "SC2"} in bond_pairs
    assert {"SC1", "SC3"} in bond_pairs
    assert len(merge_top_a.bonds) == 3

def test_merge_other_sections_shared_deduplicated(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    assert merge_top_a.other_sections["custom_section"].count("shared line") == 1

def test_merge_other_sections_unique_kept(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    lines = merge_top_a.other_sections["custom_section"]
    assert "unique line A" in lines
    assert "unique line B" in lines

def test_merge_other_section_only_in_b(merge_top_a, merge_top_b):
    merge_top_b.other_sections["only_in_b"] = ["some line"]
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    assert "only_in_b" in merge_top_a.other_sections
    assert merge_top_a.other_sections["only_in_b"] == ["some line"]

def test_merge_nondestructive_to_other(merge_top_a, merge_top_b):
    original_atoms = [a["atom"] for a in merge_top_b.atoms]
    original_bonds = len(merge_top_b.bonds)
    original_angles = len(merge_top_b.angles)
    original_vs2   = len(merge_top_b.virtual_sites2)
    merge_top_a.merge(merge_top_b)
    assert [a["atom"] for a in merge_top_b.atoms] == original_atoms
    assert len(merge_top_b.bonds)          == original_bonds
    assert len(merge_top_b.angles)         == original_angles
    assert len(merge_top_b.virtual_sites2) == original_vs2

def test_merge_fresh_indices_after(merge_top_a, merge_top_b):
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    for idx, atom in enumerate(merge_top_a.atoms, start=1):
        assert atom["nr"] == str(idx)

def test_merge_into_empty(minimal_top, merge_top_b):
    minimal_top.merge(merge_top_b, remove_redundant=True)
    atom_names = [a["atom"] for a in minimal_top.atoms]
    assert "BB"  in atom_names
    assert "SC1" in atom_names
    assert len(minimal_top.bonds) == len(merge_top_b.bonds)

def test_merge_with_empty(merge_top_a, minimal_top):
    original_atoms = [a["atom"] for a in merge_top_a.atoms]
    original_bonds = len(merge_top_a.bonds)
    merge_top_a.merge(minimal_top, remove_redundant=True)
    assert [a["atom"] for a in merge_top_a.atoms] == original_atoms
    assert len(merge_top_a.bonds) == original_bonds

def test_merge_chainable(merge_top_a, merge_top_b, minimal_top):
    result = merge_top_a.merge(merge_top_b).merge(minimal_top)
    assert result is merge_top_a
    

### Symmetry tests
def test_bond_reversed_is_duplicate(merge_top_a, merge_top_b):
    """Bond SC1-BB in B should be treated as duplicate of BB-SC1 in A."""
    # manually add reversed bond to B before merge
    merge_top_b.add_entry("bonds", "SC1", "BB", params=["1", "0.35", "2000"])
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    bb_sc1_count = sum(1 for b in merge_top_a.bonds
                       if set((b["i"], b["j"])) == {"BB", "SC1"})
    assert bb_sc1_count == 1

def test_angle_reversed_is_duplicate(merge_top_a, merge_top_b):
    """Angle SC3-SC1-BB in B should be treated as duplicate of BB-SC1-SC3 in A."""
    merge_top_b.add_entry("angles", "SC3", "SC1", "BB", params=["2", "100", "25"])
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    bb_sc1_sc3_count = sum(1 for a in merge_top_a.angles
                           if {a["i"], a["j"], a["k"]} == {"BB", "SC1", "SC3"})
    assert bb_sc1_sc3_count == 1

def test_dihedral_reversed_is_duplicate(merge_top_a, merge_top_b):
    """Dihedral SC3-SC1-BB-VS1 should be treated as duplicate of VS1-BB-SC1-SC3."""
    merge_top_a.add_entry("dihedrals", "VS1", "BB", "SC1", "SC3", params=["1", "180", "5", "1"])
    merge_top_b.add_entry("dihedrals", "SC3", "SC1", "BB", "VS1", params=["1", "180", "5", "1"])
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    dih_count = sum(1 for d in merge_top_a.dihedrals
                    if {d["i"], d["j"], d["k"], d["l"]} == {"VS1", "BB", "SC1", "SC3"})
    assert dih_count == 1

def test_exclusion_order_independent(merge_top_a, merge_top_b):
    """Exclusion SC1-BB should be treated as duplicate of BB-SC1."""
    merge_top_b.add_entry("exclusions", "SC1", "BB")
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    bb_sc1_count = sum(1 for e in merge_top_a.exclusions
                       if set(e) == {"BB", "SC1"})
    assert bb_sc1_count == 1

def test_pairs_reversed_is_duplicate(merge_top_a, merge_top_b):
    """Pair SC1-BB should be treated as duplicate of BB-SC1."""
    merge_top_a.add_entry("pairs", "BB", "SC1", params=["1"])
    merge_top_b.add_entry("pairs", "SC1", "BB", params=["1"])
    merge_top_a.merge(merge_top_b, remove_redundant=True)
    bb_sc1_count = sum(1 for p in merge_top_a.pairs
                       if set((p["i"], p["j"])) == {"BB", "SC1"})
    assert bb_sc1_count == 1


### resname tests
def test_set_resname_updates_all_atoms(simple_top):
    simple_top.set_resname("NEW")
    assert all(a["residue"] == "NEW" for a in simple_top.atoms)

def test_set_resname_updates_moleculetype(simple_top):
    simple_top.set_resname("NEW")
    assert simple_top.moleculetype["name"] == "NEW"

def test_set_resname_creates_moleculetype_if_missing():
    itp = """
[ atoms ]
1 P4 1 MOL BB 1 0.0
"""
    top = ITPTopology(itp)
    top.set_resname("NEW")
    assert top.moleculetype["name"] == "NEW"
    assert top.moleculetype["nrexcl"] == "1"

def test_set_resname_survives_roundtrip(simple_top, tmp_path):
    simple_top.set_resname("NEW")
    path = tmp_path / "out.itp"
    simple_top.write_itp(str(path))
    top2 = ITPTopology(path.read_text())
    assert all(a["residue"] == "NEW" for a in top2.atoms)
    assert top2.moleculetype["name"] == "NEW"
