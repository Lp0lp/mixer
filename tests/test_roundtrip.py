import pytest
from mixer.parser import ITPTopology

def roundtrip(top, tmp_path):
    """Helper: write and re-parse a topology."""
    path = tmp_path / "out.itp"
    top.write_itp(str(path))
    return ITPTopology(path.read_text())

def test_atom_fields_preserved(simple_top, tmp_path):
    top2 = roundtrip(simple_top, tmp_path)
    for a1, a2 in zip(simple_top.atoms, top2.atoms):
        assert a1["atom"]    == a2["atom"]
        assert a1["type"]    == a2["type"]
        assert a1["charge"]  == a2["charge"]
        assert a1["residue"] == a2["residue"]
        assert a1["resnr"]   == a2["resnr"]

def test_indices_resequenced(tmp_path):
    itp = """
[ moleculetype ]
MOL 1
[ atoms ]
5 P4  1 MOL BB  5 0.0
9 SC4 1 MOL SC1 9 0.0
"""
    top2 = roundtrip(ITPTopology(itp), tmp_path)
    assert top2.atoms[0]["nr"] == "1"
    assert top2.atoms[1]["nr"] == "2"

def test_bond_params_preserved(simple_top, tmp_path):
    top2 = roundtrip(simple_top, tmp_path)
    for b1, b2 in zip(simple_top.bonds, top2.bonds):
        assert b1["i"]      == b2["i"]
        assert b1["j"]      == b2["j"]
        assert b1["params"] == b2["params"]

def test_virtual_sitesn_survives_roundtrip(vs_top, tmp_path):
    top2 = roundtrip(vs_top, tmp_path)
    vs1 = vs_top.virtual_sitesn[0]
    vs2 = top2.virtual_sitesn[0]
    assert vs1["vs"]   == vs2["vs"]
    assert vs1["type"] == vs2["type"]
    assert vs1["refs"] == vs2["refs"]

def test_virtual_sites2_survives_roundtrip(vs_top, tmp_path):
    top2 = roundtrip(vs_top, tmp_path)
    vs1 = vs_top.virtual_sites2[0]
    vs2 = top2.virtual_sites2[0]
    assert vs1["i"] == vs2["i"]
    assert vs1["j"] == vs2["j"]
    assert vs1["k"] == vs2["k"]
    assert vs1["params"] == vs2["params"]

def test_exclusions_survive_roundtrip(simple_top, tmp_path):
    top2 = roundtrip(simple_top, tmp_path)
    assert simple_top.exclusions[0] == top2.exclusions[0]

def test_moleculetype_survives_roundtrip(simple_top, tmp_path):
    top2 = roundtrip(simple_top, tmp_path)
    assert simple_top.moleculetype["name"]   == top2.moleculetype["name"]
    assert simple_top.moleculetype["nrexcl"] == top2.moleculetype["nrexcl"]
    
def test_empty_sections_not_written(simple_top, tmp_path):
    path = tmp_path / "out.itp"
    simple_top.write_itp(str(path))
    content = path.read_text()
    assert "[ dihedrals ]" not in content
    assert "[ virtual_sites2 ]" not in content

def test_unknown_sections_preserved(tmp_path):
    itp = """
[ moleculetype ]
MOL 1
[ atoms ]
1 P4 1 MOL BB 1 0.0
[ unknown_section ]
some raw line
"""
    top2 = roundtrip(ITPTopology(itp), tmp_path)
    assert "unknown_section" in top2.other_sections
    assert top2.other_sections["unknown_section"] == ["some raw line"]

def test_section_order(simple_top, tmp_path):
    path = tmp_path / "out.itp"
    simple_top.write_itp(str(path))
    content = path.read_text()
    sections = [l.strip("[] \n") for l in content.splitlines() if l.startswith("[")]
    expected = [s for s in ITPTopology.STRUCTURED_SECTIONS if s in sections]
    assert sections == expected

def test_add_atom_survives_roundtrip(simple_top, tmp_path):
    simple_top.add_atom("SC3", "SC4", "0.0")
    top2 = roundtrip(simple_top, tmp_path)
    assert len(top2.atoms) == 4
    assert top2.atoms[-1]["atom"] == "SC3"
    assert top2.atoms[-1]["nr"] == "4"

def test_remove_atom_survives_roundtrip(simple_top, tmp_path):
    simple_top.remove_atom("SC1")
    top2 = roundtrip(simple_top, tmp_path)
    atom_names = [a["atom"] for a in top2.atoms]
    assert "SC1" not in atom_names
    assert top2.atoms[0]["nr"] == "1"
    assert top2.atoms[1]["nr"] == "2"

def test_add_bond_survives_roundtrip(simple_top, tmp_path):
    simple_top.add_atom("SC3", "SC4", "0.0")
    simple_top.add_entry("bonds", "SC2", "SC3", params=["1", "0.47", "3800"])
    top2 = roundtrip(simple_top, tmp_path)
    assert len(top2.bonds) == 3
    assert top2.bonds[-1]["i"] == "SC2"
    assert top2.bonds[-1]["j"] == "SC3"
    assert top2.bonds[-1]["params"] == ["1", "0.47", "3800"]

def test_atom_mass_survives_roundtrip(vs_top, tmp_path):
    top2 = roundtrip(vs_top, tmp_path)
    for a1, a2 in zip(vs_top.atoms, top2.atoms):
        assert a1["mass"] == a2["mass"]
        
def test_write_itp_idempotent(simple_top, tmp_path):
    """Writing twice should produce identical output."""
    path1 = tmp_path / "out1.itp"
    path2 = tmp_path / "out2.itp"
    simple_top.write_itp(str(path1))
    simple_top.write_itp(str(path2))
    assert path1.read_text() == path2.read_text()

def test_from_file(simple_top, tmp_path):
    """from_file should produce identical topology to from string."""
    path = tmp_path / "out.itp"
    simple_top.write_itp(str(path))
    top2 = ITPTopology.from_file(str(path))
    assert [a["atom"] for a in simple_top.atoms] == [a["atom"] for a in top2.atoms]
    assert len(simple_top.bonds) == len(top2.bonds)