import re
import pytest

from mixer.acyl import (
    _parse_spec,
    _validate_tail_string,
    _tail_bond_name,
    build_tail_itp,
    TAIL_CODES,
    linkerMapp,
    tailMapp,
    CHAIN_INDEX,
)


# Helpers
def _atoms(itp_text):
    """Return list of atom-name strings from the [ atoms ] section of an itp string."""
    in_atoms = False
    names = []
    for line in itp_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[ atoms ]"):
            in_atoms = True
            continue
        if in_atoms:
            if stripped.startswith("["):
                break
            if not stripped or stripped.startswith(";"):
                continue
            names.append(stripped.split()[4])  # col 4 = atom name
    return names


def _bonds(itp_text):
    """Return list of (i, j, name) tuples from the [ bonds ] section."""
    in_bonds = False
    result = []
    for line in itp_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[ bonds ]"):
            in_bonds = True
            continue
        if in_bonds:
            if stripped.startswith("["):
                break
            if not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            result.append((int(parts[0]), int(parts[1]), parts[2]))
    return result


def _angles(itp_text):
    """Return list of (i, j, k, name) tuples from the [ angles ] section."""
    in_angles = False
    result = []
    for line in itp_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[ angles ]"):
            in_angles = True
            continue
        if in_angles:
            if stripped.startswith("["):
                break
            if not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            result.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
    return result


# _parse_spec
class TestParseSpec:

    def test_linker_explicit_tail(self):
        linker, tail = _parse_spec("G-cC")
        assert linker == "G"
        assert tail == "cC"

    def test_linker_shorthand_tail_expanded(self):
        # T → cC
        linker, tail = _parse_spec("G-T")
        assert linker == "G"
        assert tail == TAIL_CODES["T"]

    def test_no_linker_explicit_tail(self):
        linker, tail = _parse_spec("cC")
        assert linker is None
        assert tail == "cC"

    def test_no_linker_shorthand_expanded(self):
        linker, tail = _parse_spec("O")
        assert linker is None
        assert tail == TAIL_CODES["O"]

    def test_multichar_linker(self):
        linker, tail = _parse_spec("A1-tCCC")
        assert linker == "A1"
        assert tail == "tCCC"

    def test_dha_shorthand_expanded(self):
        # DHA is a 3-char key — make sure it doesn't split on the first char
        linker, tail = _parse_spec("DHA")
        assert linker is None
        assert tail == TAIL_CODES["DHA"]

    def test_empty_spec_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            _parse_spec("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            _parse_spec("   ")

    def test_unknown_linker_raises(self):
        with pytest.raises(ValueError, match="Unsupported linker"):
            _parse_spec("ZZ-CC")

    def test_missing_tail_after_dash_raises(self):
        with pytest.raises(ValueError, match="Missing tail"):
            _parse_spec("G-")

    def test_leading_trailing_whitespace_stripped(self):
        linker, tail = _parse_spec("  G-cC  ")
        assert linker == "G"
        assert tail == "cC"

    @pytest.mark.parametrize("linker_key", list(linkerMapp.keys()))
    def test_all_linkers_accepted(self, linker_key):
        linker, tail = _parse_spec(f"{linker_key}-CC")
        assert linker == linker_key

    @pytest.mark.parametrize("code,pattern", list(TAIL_CODES.items()))
    def test_all_shorthand_codes_expand(self, code, pattern):
        _, tail = _parse_spec(code)
        assert tail == pattern


# _validate_tail_string
class TestValidateTailString:

    def test_valid_passes(self):
        _validate_tail_string("cCDCC")  # no exception

    def test_invalid_char_raises(self):
        with pytest.raises(ValueError, match="Unsupported tail code"):
            _validate_tail_string("cCXCC")

    def test_all_valid_tail_codes(self):
        _validate_tail_string("".join(tailMapp.keys()))



# _tail_bond_name
class TestTailBondName:
    """
    Rule summary:
      pos==2, both non-small  → b_<prev>_<curr>_mid_5long
      pos==tail_len           → b_<prev>_<curr>_end
      otherwise               → b_<prev>_<curr>_mid

    Small beads: bead type starts with "S" (e.g. SC1 for 'c').
    Regular beads: C1 for 'C', C4h for 'D', C5h for 'F', etc.
    """

    # pos=2, both regular-sized → _mid_5long
    def test_pos2_both_regular_gives_mid_5long(self):
        # C→C: both C1 (not small), pos=2
        result = _tail_bond_name("C", "C", bead_pos=2, tail_len=4)
        assert result == "b_C1_C1_mid_5long"

    # pos=2, prev is small → _mid (not 5long)
    def test_pos2_prev_small_gives_mid(self):
        # c→C: prev is SC1 (small), curr is C1
        result = _tail_bond_name("c", "C", bead_pos=2, tail_len=4)
        assert result == "b_SC1_C1_mid"

    # pos=2, curr is small → _mid (not 5long)
    def test_pos2_curr_small_gives_mid(self):
        result = _tail_bond_name("C", "c", bead_pos=2, tail_len=4)
        assert result == "b_C1_SC1_mid"

    # pos=2, both small → _mid (not 5long)
    def test_pos2_both_small_gives_mid(self):
        result = _tail_bond_name("c", "c", bead_pos=2, tail_len=4)
        assert result == "b_SC1_SC1_mid"

    # last bead → _end (even if pos==2 for a 2-bead tail)
    def test_last_bond_gives_end(self):
        # pos==2 check doesn't fire here, so _end wins
        result = _tail_bond_name("C", "C", bead_pos=4, tail_len=4)
        assert result == "b_C1_C1_end"

    # a 2-bead tail: pos=2 IS the last bond — _mid_5long wins over _end
    def test_two_bead_tail_pos2_wins_over_end(self):
        # pos==2 check fires before tail_len check, so _mid_5long wins even though
        # this is also the last bond
        result = _tail_bond_name("C", "C", bead_pos=2, tail_len=2)
        assert result == "b_C1_C1_mid_5long"

    # middle bond, not pos=2 → _mid
    def test_middle_bond_gives_mid(self):
        result = _tail_bond_name("C", "C", bead_pos=3, tail_len=5)
        assert result == "b_C1_C1_mid"

    # unsaturated beads use their bond codes
    def test_unsaturated_bead_codes(self):
        # D has bond code C4
        result = _tail_bond_name("D", "C", bead_pos=3, tail_len=4)
        assert result == "b_C4_C1_mid"

    def test_end_uses_correct_codes_for_mixed(self):
        result = _tail_bond_name("D", "C", bead_pos=4, tail_len=4)
        assert result == "b_C4_C1_end"


# build_tail_itp — atom names and counts
class TestBuildTailItpAtoms:

    def test_tail_only_atom_count(self):
        itp = build_tail_itp("cC")
        assert len(_atoms(itp)) == 2

    def test_linker_plus_tail_atom_count(self):
        # G linker + cC tail = 3 atoms
        itp = build_tail_itp("G-cC")
        assert len(_atoms(itp)) == 3

    def test_shorthand_same_as_explicit(self):
        # T expands to cC
        itp_short = build_tail_itp("G-T", chain_id="A")
        itp_explicit = build_tail_itp("G-cC", chain_id="A")
        assert _atoms(itp_short) == _atoms(itp_explicit)

    def test_linker_atom_name_chain_a(self):
        itp = build_tail_itp("G-cC", chain_id="A")
        atoms = _atoms(itp)
        assert atoms[0] == "GL1"   # chain A → index 1

    def test_linker_atom_name_chain_b(self):
        itp = build_tail_itp("G-cC", chain_id="B")
        atoms = _atoms(itp)
        assert atoms[0] == "GL2"   # chain B → index 2

    def test_tail_bead_names_chain_a(self):
        itp = build_tail_itp("G-cC", chain_id="A")
        atoms = _atoms(itp)
        # tail beads: C1A (bead 1, chain A), C2A (bead 2, chain A)
        assert atoms[1] == "C1A"
        assert atoms[2] == "C2A"

    def test_tail_bead_names_chain_b(self):
        itp = build_tail_itp("G-cC", chain_id="B")
        atoms = _atoms(itp)
        assert atoms[1] == "C1B"
        assert atoms[2] == "C2B"

    def test_tail_only_no_linker_bead(self):
        itp = build_tail_itp("CC")
        atoms = _atoms(itp)
        assert atoms[0] == "C1A"
        assert atoms[1] == "C2A"

    def test_chain_id_case_insensitive(self):
        itp_upper = build_tail_itp("G-cC", chain_id="A")
        itp_lower = build_tail_itp("G-cC", chain_id="a")
        assert _atoms(itp_upper) == _atoms(itp_lower)

    def test_invalid_chain_id_raises(self):
        with pytest.raises(ValueError):
            build_tail_itp("G-cC", chain_id="1")

    def test_fa_linker_no_question_mark_in_name(self):
        # FA linker bead name is "COO", no "?" — should not crash
        itp = build_tail_itp("FA-cC")
        atoms = _atoms(itp)
        assert atoms[0] == "COO"

    def test_custom_resname_appears_in_atoms(self):
        itp = build_tail_itp("G-cC", resname="MYRES")
        assert "MYRES" in itp

    def test_custom_molname_appears_in_moleculetype(self):
        itp = build_tail_itp("G-cC", molname="MYMOL")
        assert "MYMOL" in itp

    def test_default_molname_derived_from_spec(self):
        itp = build_tail_itp("G-cC")
        assert "G_cC" in itp

    @pytest.mark.parametrize("code,pattern", list(TAIL_CODES.items()))
    def test_all_shorthand_codes_build_without_error(self, code, pattern):
        # Just check it doesn't raise and has the right bead count
        itp = build_tail_itp(f"G-{code}")
        expected_atoms = 1 + len(pattern)  # 1 linker + tail length
        assert len(_atoms(itp)) == expected_atoms



# build_tail_itp — bonds
class TestBuildTailItpBonds:

    def test_bond_count_equals_atoms_minus_one(self):
        # for any connected chain: bonds = atoms - 1
        for spec in ["cC", "G-cC", "G-CDCC", "E-O"]:
            itp = build_tail_itp(spec)
            n_atoms = len(_atoms(itp))
            n_bonds = len(_bonds(itp))
            assert n_bonds == n_atoms - 1, f"Failed for spec={spec!r}"

    def test_tail_only_no_linker_bond(self):
        # single tail with 2 beads: 1 bond, no linker involved
        bonds = _bonds(build_tail_itp("CC"))
        assert len(bonds) == 1
        assert bonds[0][0] == 1
        assert bonds[0][1] == 2

    def test_linker_to_tail_bond_is_first(self):
        bonds = _bonds(build_tail_itp("G-CC"))
        assert bonds[0][0] == 1   # linker
        assert bonds[0][1] == 2   # first tail bead

    def test_linker_to_regular_tail_bond_name_has_5long(self):
        # C tail bead is non-small → bond name gets _5long suffix
        bonds = _bonds(build_tail_itp("G-CC"))
        assert bonds[0][2].endswith("_5long")

    def test_linker_to_small_tail_bond_name_no_5long(self):
        # c tail bead is small (SC1) → no _5long
        bonds = _bonds(build_tail_itp("G-cC"))
        assert not bonds[0][2].endswith("_5long")

    def test_second_internal_bond_is_mid_5long_both_regular(self):
        # CC[C]: bond between bead 2 and 3 is pos=2, both C1 → _mid_5long
        bonds = _bonds(build_tail_itp("CCC"))
        # bonds[0] = 1→2 (pos=2 in tail), bonds[1] = 2→3 (pos=3, last)
        assert "_mid_5long" in bonds[0][2]

    def test_last_bond_is_end(self):
        bonds = _bonds(build_tail_itp("CCC"))
        assert bonds[-1][2].endswith("_end")

    def test_middle_bond_is_mid(self):
        # 4-bead tail: bonds at pos 2 (mid_5long), 3 (mid), 4 (end)
        bonds = _bonds(build_tail_itp("CCCC"))
        assert "_mid" in bonds[1][2] and "_5long" not in bonds[1][2]

    def test_single_bead_tail_no_bonds(self):
        # 1-bead tail with no linker: no bonds at all
        bonds = _bonds(build_tail_itp("C"))
        assert bonds == []

    def test_linker_plus_single_bead_one_bond(self):
        bonds = _bonds(build_tail_itp("G-C"))
        assert len(bonds) == 1


# build_tail_itp — angles
class TestBuildTailItpAngles:

    def test_no_angles_for_two_bead_tail_no_linker(self):
        itp = build_tail_itp("CC")
        assert "[ angles ]" not in itp

    def test_no_angles_for_one_linker_one_tail_bead(self):
        itp = build_tail_itp("G-C")
        assert "[ angles ]" not in itp

    def test_linker_plus_two_tail_beads_has_one_linker_angle(self):
        # G + CC: one linker-tail-tail angle, no internal angles
        angles = _angles(build_tail_itp("G-CC"))
        assert len(angles) == 1
        assert angles[0][0] == 1  # linker is atom 1

    def test_three_bead_tail_has_one_internal_angle(self):
        angles = _angles(build_tail_itp("CCC"))
        assert len(angles) == 1

    def test_linker_plus_three_bead_tail_angle_count(self):
        # G + CCC: 1 linker-tail-tail + 1 internal = 2
        angles = _angles(build_tail_itp("G-CCC"))
        assert len(angles) == 2

    def test_angle_count_formula(self):
        # n-bead tail (no linker): angles = max(0, n-2)
        for tail, expected in [("C", 0), ("CC", 0), ("CCC", 1), ("CCCC", 2), ("CCCCC", 3)]:
            itp = build_tail_itp(tail)
            assert len(_angles(itp)) == expected, f"tail={tail!r}"

    def test_linker_angle_name_format(self):
        # G linker, C tail: angle name should follow a_GL_C1_C1_glyc
        angles = _angles(build_tail_itp("G-CC"))
        name = angles[0][3]
        assert name.startswith("a_GL_")
        assert name.endswith("_glyc")

    def test_internal_angle_name_ends_with_def(self):
        angles = _angles(build_tail_itp("CCC"))
        assert angles[0][3].endswith("_def")

    def test_unsaturated_tail_angle_uses_correct_codes(self):
        # D bead has angle code C4, C has C1
        # CDC: angles — C-D-C → a_C1_C4_C1_def
        angles = _angles(build_tail_itp("CDC"))
        assert angles[0][3] == "a_C1_C4_C1_def"


# build_tail_itp — file output
class TestBuildTailItpFileOutput:

    def test_file_written_when_out_given(self, tmp_path):
        outfile = tmp_path / "tail.itp"
        build_tail_itp("G-cC", out=str(outfile))
        assert outfile.exists()

    def test_file_content_matches_return_value(self, tmp_path):
        outfile = tmp_path / "tail.itp"
        returned = build_tail_itp("G-cC", out=str(outfile))
        assert outfile.read_text() == returned

    def test_no_file_written_when_out_is_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        build_tail_itp("G-cC")
        assert list(tmp_path.glob("*.itp")) == []