"""
Integration tests for merger.py
--------------------------------
Four fragments are merged into a single 16-atom topology.

  FRAG_A — ring headgroup (6 beads)
      A B C D  — constrained ring
      E        — virtual_sitesn (frame A-B-C-D)
      F        — virtual_sites3
      Sections: constraints (×4), bonds (×1), virtual_sitesn, virtual_sites3,
                dihedrals (×1), exclusions (×2)

  FRAG_B — glycerol backbone (6 beads)
      B C      — shared with FRAG_A → deduplicated (4 new atoms)
      GL1 GL2  — linker beads
      C1A C1B  — first tail beads, already in place
      Sections: constraints (×1, B-C deduped), bonds (×2), angles (×4), dihedrals (×1)

  TAIL_A — oleoyl off GL1, chain A  (G-O → CDCC)
      GL1 C1A D2A C3A C4A
      GL1 and C1A deduplicated → 3 new atoms (D2A C3A C4A)

  TAIL_B — linoleoyl off GL2, chain B  (G-L → CDDC)
      GL2 C1B D2B D3B C4B
      GL2 and C1B deduplicated → 3 new atoms (D2B D3B C4B)

Atom deduplication table
------------------------
  FRAG_A:  A B C D E F                        →  6 atoms
  FRAG_B:  B* C* GL1 GL2 C1A C1B             →  4 new  (B, C deduped)
  TAIL_A:  GL1* C1A* D2A C3A C4A             →  3 new  (GL1, C1A deduped)
  TAIL_B:  GL2* C1B* D2B D3B C4B             →  3 new  (GL2, C1B deduped)
  TOTAL:   16

Bond deduplication
------------------
  Bond symmetry keys are (min,max) pairs. A bond is only deduped if the
  *exact same pair* appeared in a prior fragment.

  FRAG_A : A-C                                              → 1
  FRAG_B : C-GL1, GL1-GL2                                  → 2
  TAIL_A : GL1-C1A, C1A-D2A, D2A-C3A, C3A-C4A             → 4
  TAIL_B : GL2-C1B, C1B-D2B, D2B-D3B, D3B-C4B             → 4
  TOTAL  : 11

Angle deduplication
-------------------
  FRAG_B : B-C-GL1, C-GL1-GL2, GL1-GL2-C1A, GL2-GL1-C1B  → 4
  TAIL_A : GL1-C1A-D2A, C1A-D2A-C3A, D2A-C3A-C4A         → 3
  TAIL_B : GL2-C1B-D2B, C1B-D2B-D3B, D2B-D3B-C4B         → 3
  TOTAL  : 10

All counts verified against the actual merge output before writing these tests.
"""

import textwrap
import pytest
from pathlib import Path

from mixer.acyl import build_tail_itp
from mixer.merger import merge_itps


# Fragment itps
FRAG_A_ITP = textwrap.dedent("""\
    [ moleculetype ]
    A  1
    [ atoms ]
    ; nr type resnr residue atom cgnr charge mass
       1    SN3a       0    UNK     A         1       0
       2    TN1        0    UNK     B         2       0
       3    TC5        0    UNK     C         3       0
       4    TQ4p       0    UNK     D         4       1
       5    SP2        0    UNK     E         5       0    0
       6    SC1        0    UNK     F         6       0    0
    [ constraints ]
       1       2       1    0.30    5000
       2       3       1    0.30    5000
       3       4       1    0.30    5000
       4       1       1    0.30    5000
    [ bonds ]
        1   3    1   0.4243   5000
    [ virtual_sitesn ]
    5  1  1 2 3 4
    [ virtual_sites3 ]
     6   1 2 3   0.3   0.3 0.3
    [ dihedrals ]
       1 2 3 4      2     0       250
    [ exclusions ]
    5 1 2 3 4 6
    6 1 2 3 4 5
""")

FRAG_B_ITP = textwrap.dedent("""\
    [ moleculetype ]
    B  1
    [ atoms ]
    ; nr type resnr residue atom cgnr charge mass
       1    TN2        0    UNK     B         1       0
       2    TC5        0    UNK     C         2       0
       3    SN4a       0    UNK     GL1       3       0
       4    SN4a       0    UNK     GL2       4       0
       5    C1         0    UNK     C1A       5       0
       6    C1         0    UNK     C1B       6       0
    [ constraints ]
       1       2       1    0.30    5000
    [ bonds ]
        2   3    1   0.4243   5000
        3   4    b_GL_GL_glyc
    [ angles ]
    1 2 3   1  90.  20
    2 3 4  a_PO4_GL_C_def
    3 4 5  a_GL_GL_C_glyc
    4 3 6  a_GL_GL_C_glyc
    [ dihedrals ]
    1 2 3 4  1  0  20  1
""")


# Fixtures
@pytest.fixture
def frag_a(tmp_path):
    p = tmp_path / "frag_a.itp"
    p.write_text(FRAG_A_ITP)
    return p


@pytest.fixture
def frag_b(tmp_path):
    p = tmp_path / "frag_b.itp"
    p.write_text(FRAG_B_ITP)
    return p


@pytest.fixture
def tail_a(tmp_path):
    """Oleoyl off GL1, chain A (G-O → CDCC): GL1 C1A D2A C3A C4A."""
    p = tmp_path / "tail_a.itp"
    p.write_text(build_tail_itp("G-O", chain_id="A", resname="UNK", molname="TAIL_A"))
    return p


@pytest.fixture
def tail_b(tmp_path):
    """Linoleoyl off GL2, chain B (G-L → CDDC): GL2 C1B D2B D3B C4B."""
    p = tmp_path / "tail_b.itp"
    p.write_text(build_tail_itp("G-L", chain_id="B", resname="UNK", molname="TAIL_B"))
    return p


@pytest.fixture
def merged(tmp_path, frag_a, frag_b, tail_a, tail_b):
    return merge_itps(
        [frag_a, frag_b, tail_a, tail_b],
        resname="MOL", outdir=tmp_path, author="Mixer",)


# Helpers
def _section(text, name):
    """Non-blank, non-comment lines from a named [ section ]."""
    in_sec = False
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s == f"[ {name} ]":
            in_sec = True
            continue
        if in_sec:
            if s.startswith("["):
                break
            if s and not s.startswith(";"):
                lines.append(s)
    return lines


def _atom_names(text):
    return [l.split()[4] for l in _section(text, "atoms")]


def _atom_types(text):
    return {l.split()[4]: l.split()[1] for l in _section(text, "atoms")}


def _atom_charges(text):
    return {l.split()[4]: l.split()[6] for l in _section(text, "atoms")}


def _bond_pairs(text):
    return [(int(l.split()[0]), int(l.split()[1])) for l in _section(text, "bonds")]


def _bond_names(text):
    return [" ".join(l.split()[2:]) for l in _section(text, "bonds")]


def _angle_triples(text):
    return [(int(l.split()[0]), int(l.split()[1]), int(l.split()[2]))
            for l in _section(text, "angles")]


def _constraint_pairs(text):
    return [(int(l.split()[0]), int(l.split()[1])) for l in _section(text, "constraints")]


def _dihedral_quads(text):
    return [(int(l.split()[0]), int(l.split()[1]),
             int(l.split()[2]), int(l.split()[3]))
            for l in _section(text, "dihedrals")]


def _exclusion_sets(text):
    return [frozenset(int(x) for x in l.split())
            for l in _section(text, "exclusions")]


def _vsn_lines(text):
    return _section(text, "virtual_sitesn")


def _vs3_lines(text):
    return _section(text, "virtual_sites3")


# Guard clauses
class TestGuards:

    def test_empty_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No input"):
            merge_itps([], outdir=tmp_path)

    def test_redundancy_priority_wrong_length_raises(self, tmp_path, frag_a, frag_b):
        with pytest.raises(ValueError, match="redundancy_priority"):
            merge_itps([frag_a, frag_b], redundancy_priority=[True], outdir=tmp_path)

    def test_redundancy_priority_correct_length_accepted(self, tmp_path, frag_a, frag_b):
        merge_itps([frag_a, frag_b],
                   redundancy_priority=[False, False],
                   resname="MOL", outdir=tmp_path)


# Output file
class TestOutputFile:

    def test_file_exists(self, merged):
        assert merged.exists()

    def test_filename_matches_resname(self, merged):
        assert merged.name == "MOL.itp"

    def test_outdir_created_when_missing(self, tmp_path, frag_a, frag_b):
        new_dir = tmp_path / "deep" / "nested"
        assert not new_dir.exists()
        merge_itps([frag_a, frag_b], resname="MOL", outdir=new_dir)
        assert new_dir.exists()


# Header
class TestHeader:

    def test_starts_with_triple_semicolon(self, merged):
        assert merged.read_text().startswith(";;;")

    def test_author_in_header(self, merged):
        assert "Mixer" in merged.read_text()

    def test_all_fragment_filenames_in_header(self, merged, frag_a, frag_b, tail_a, tail_b):
        text = merged.read_text()
        for frag in (frag_a, frag_b, tail_a, tail_b):
            assert frag.name in text

    def test_custom_author_written(self, tmp_path, frag_a, frag_b):
        out = merge_itps([frag_a, frag_b], resname="MOL",
                         outdir=tmp_path, author="TestSuite")
        assert "TestSuite" in out.read_text()


# Atoms
class TestAtoms:

    EXPECTED_ATOMS = [
        "A", "B", "C", "D", "E", "F",      # frag_a ring
        "GL1", "GL2", "C1A", "C1B",         # frag_b (B, C deduped)
        "D2A", "C3A", "C4A",                # tail_a (GL1, C1A deduped)
        "D2B", "D3B", "C4B",                # tail_b (GL2, C1B deduped)
    ]

    def test_total_atom_count(self, merged):
        assert len(_atom_names(merged.read_text())) == 16

    def test_no_duplicate_atom_names(self, merged):
        names = _atom_names(merged.read_text())
        assert len(names) == len(set(names))

    def test_all_expected_atoms_present(self, merged):
        names = set(_atom_names(merged.read_text()))
        for atom in self.EXPECTED_ATOMS:
            assert atom in names, f"Expected atom {atom!r} missing"

    def test_indices_sequential_from_one(self, merged):
        indices = [int(l.split()[0]) for l in _section(merged.read_text(), "atoms")]
        assert indices == list(range(1, 17))

    def test_resname_applied_to_all_atoms(self, merged):
        for line in _section(merged.read_text(), "atoms"):
            assert line.split()[3] == "MOL"

    def test_moleculetype_name_is_resname(self, merged):
        mol_lines = _section(merged.read_text(), "moleculetype")
        assert mol_lines[0].split()[0] == "MOL"

    def test_frag_a_bead_types_preserved(self, merged):
        types = _atom_types(merged.read_text())
        assert types["A"] == "SN3a"
        assert types["B"] == "TN1"    # frag_a wins as base
        assert types["C"] == "TC5"
        assert types["D"] == "TQ4p"
        assert types["E"] == "SP2"
        assert types["F"] == "SC1"

    def test_glycerol_bead_types_preserved(self, merged):
        types = _atom_types(merged.read_text())
        assert types["GL1"] == "SN4a"
        assert types["GL2"] == "SN4a"

    def test_oleoyl_bead_types(self, merged):
        types = _atom_types(merged.read_text())
        assert types["C1A"] == "C1"    # regular C1 — first bead of CDCC
        assert types["D2A"] == "C4h"   # D bead
        assert types["C3A"] == "C1"
        assert types["C4A"] == "C1"

    def test_linoleoyl_bead_types(self, merged):
        types = _atom_types(merged.read_text())
        assert types["C1B"] == "C1"    # regular C1 — first bead of CDDC
        assert types["D2B"] == "C4h"
        assert types["D3B"] == "C4h"
        assert types["C4B"] == "C1"

    def test_charged_atom_D_preserved(self, merged):
        charges = _atom_charges(merged.read_text())
        assert charges["D"] == "1"


# Bonds
class TestBonds:

    def test_bond_count(self, merged):
        # frag_a: 1  (A-C)
        # frag_b: 2  (C-GL1, GL1-GL2)
        # tail_a: 4  (GL1-C1A, C1A-D2A, D2A-C3A, C3A-C4A)
        # tail_b: 4  (GL2-C1B, C1B-D2B, D2B-D3B, D3B-C4B)
        # total:  11
        assert len(_bond_pairs(merged.read_text())) == 11

    def test_bond_indices_in_range(self, merged):
        text = merged.read_text()
        n = len(_atom_names(text))
        for i, j in _bond_pairs(text):
            assert 1 <= i <= n
            assert 1 <= j <= n

    def test_both_gl_c1_bonds_have_5long(self, merged):
        # C1A and C1B are regular C1 beads → both linker bonds get _5long
        bond_names = _bond_names(merged.read_text())
        gl_c1_5long = [b for b in bond_names if "GL" in b and "C1" in b and "5long" in b]
        assert len(gl_c1_5long) == 2

    def test_oleoyl_second_bond_mid_5long(self, merged):
        # C1A-D2A: pos=2, both regular → _mid_5long
        bond_names = _bond_names(merged.read_text())
        assert any("C1_C4_mid_5long" in b for b in bond_names)

    def test_oleoyl_terminal_bond_is_end(self, merged):
        bond_names = _bond_names(merged.read_text())
        assert any("C1_C1_end" in b for b in bond_names)

    def test_linoleoyl_terminal_bond_is_end(self, merged):
        bond_names = _bond_names(merged.read_text())
        assert any("C4_C1_end" in b for b in bond_names)


# Constraints
class TestConstraints:

    def test_constraint_count(self, merged):
        # frag_a: 4 ring constraints (A-B, B-C, C-D, D-A)
        # frag_b: 1 (B-C) → deduplicated
        # total:  4
        assert len(_constraint_pairs(merged.read_text())) == 4

    def test_ring_constraints_present(self, merged):
        text = merged.read_text()
        names = _atom_names(text)
        pairs = {(min(i, j), max(i, j)) for i, j in _constraint_pairs(text)}
        def idx(name): return names.index(name) + 1
        assert (idx("A"), idx("B")) in pairs
        assert (idx("B"), idx("C")) in pairs
        assert (idx("C"), idx("D")) in pairs
        assert (min(idx("D"), idx("A")), max(idx("D"), idx("A"))) in pairs


# Angles
class TestAngles:

    def test_angle_count(self, merged):
        # frag_b: 4  (B-C-GL1, C-GL1-GL2, GL1-GL2-C1A, GL2-GL1-C1B)
        # tail_a: 3  (GL1-C1A-D2A, C1A-D2A-C3A, D2A-C3A-C4A)
        # tail_b: 3  (GL2-C1B-D2B, C1B-D2B-D3B, D2B-D3B-C4B)
        # total:  10
        assert len(_angle_triples(merged.read_text())) == 10

    def test_angle_indices_in_range(self, merged):
        text = merged.read_text()
        n = len(_atom_names(text))
        for i, j, k in _angle_triples(text):
            assert 1 <= i <= n
            assert 1 <= j <= n
            assert 1 <= k <= n


# Dihedrals
class TestDihedrals:

    def test_dihedral_count(self, merged):
        # frag_a: 1 (A-B-C-D), frag_b: 1 (B-C-GL1-GL2)
        assert len(_dihedral_quads(merged.read_text())) == 2

    def test_ring_dihedral_present(self, merged):
        text = merged.read_text()
        names = _atom_names(text)
        def idx(name): return names.index(name) + 1
        assert (idx("A"), idx("B"), idx("C"), idx("D")) in _dihedral_quads(text)

    def test_backbone_dihedral_present(self, merged):
        text = merged.read_text()
        names = _atom_names(text)
        def idx(name): return names.index(name) + 1
        assert (idx("B"), idx("C"), idx("GL1"), idx("GL2")) in _dihedral_quads(text)


# Virtual sites
class TestVirtualSites:

    def test_virtual_sitesn_present(self, merged):
        assert _vsn_lines(merged.read_text()), "[ virtual_sitesn ] section missing"

    def test_virtual_sites3_present(self, merged):
        assert _vs3_lines(merged.read_text()), "[ virtual_sites3 ] section missing"

    def test_virtual_sitesn_points_to_E(self, merged):
        text = merged.read_text()
        names = _atom_names(text)
        vs_idx = int(_vsn_lines(text)[0].split()[0])
        assert names[vs_idx - 1] == "E"

    def test_virtual_sites3_points_to_F(self, merged):
        text = merged.read_text()
        names = _atom_names(text)
        vs_idx = int(_vs3_lines(text)[0].split()[0])
        assert names[vs_idx - 1] == "F"


# Exclusions
class TestExclusions:

    def test_exclusion_count(self, merged):
        assert len(_exclusion_sets(merged.read_text())) == 2

    def test_exclusion_indices_in_range(self, merged):
        text = merged.read_text()
        n = len(_atom_names(text))
        for group in _exclusion_sets(text):
            for idx in group:
                assert 1 <= idx <= n