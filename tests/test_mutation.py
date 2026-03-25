from mixer.parser import ITPTopology
import pytest

class TestMutation:
    def test_add_atom_updates_numbering(self, simple_top):
        simple_top.add_atom("SC3", "Type", "0.0")
        assert len(simple_top.atoms) == 4
        assert simple_top.atoms[-1]["nr"] == "4"
    
    def test_add_atom_updates_maps(self, simple_top):
        simple_top.add_atom("SC3", "Type", "0.0")
        assert "SC3" in simple_top.name_to_nr
        assert simple_top.nr_to_name[simple_top.name_to_nr["SC3"]] == "SC3"
    
    def test_add_entry_bond_appears(self, simple_top):
        simple_top.add_atom("SC3", "Type", "0.0")
        simple_top.add_entry("bonds", "SC2", "SC3", params=["1", "0.47", "3800"])
        assert len(simple_top.bonds) == 3
        assert simple_top.bonds[-1]["i"] == "SC2"
        assert simple_top.bonds[-1]["j"] == "SC3"
    
    def test_remove_atom_updates_numbering(self, simple_top):
        simple_top.remove_atom("SC1")
        assert len(simple_top.atoms) == 2
        assert simple_top.atoms[0]["nr"] == "1"
        assert simple_top.atoms[1]["nr"] == "2"
    
    def test_remove_atom_updates_maps(self, simple_top):
        simple_top.remove_atom("SC1")
        assert "SC1" not in simple_top.name_to_nr
        assert "SC1" not in simple_top.nr_to_name.values()
    
    def test_remove_atom_cascades_bonds(self, simple_top):
        simple_top.remove_atom("SC1")
        names_in_bonds = [n for b in simple_top.bonds for n in (b["i"], b["j"])]
        assert "SC1" not in names_in_bonds
    
    def test_remove_atom_cascades_angles(self, simple_top):
        simple_top.remove_atom("SC1")
        names_in_angles = [n for a in simple_top.angles for n in (a["i"], a["j"], a["k"])]
        assert "SC1" not in names_in_angles

    def test_add_atom_duplicate_raises(self, simple_top):
        with pytest.raises(ValueError, match="already exists"):
            simple_top.add_atom("BB", "P4", "0.0")
    
    def test_add_atom_inherits_residue(self, simple_top):
        simple_top.add_atom("SC3", "SC4", "0.0")
        new = simple_top.atoms[-1]
        assert new["resnr"] == simple_top.atoms[0]["resnr"]
        assert new["residue"] == simple_top.atoms[0]["residue"]
    
    def test_add_atom_empty_topology(self, minimal_top):
        minimal_top.add_atom("SC1", "SC4", "0.0")
        assert minimal_top.atoms[-1]["residue"] == "MIN"
        assert minimal_top.atoms[-1]["resnr"] == "1"
    
    def test_add_entry_errors(self, simple_top):
        with pytest.raises(ValueError, match="atoms"):
            simple_top.add_entry("atoms", "BB")
        with pytest.raises(ValueError, match="Unknown"):
            simple_top.add_entry("garbage", "BB", "SC1")
        with pytest.raises(ValueError, match="expects"):
            simple_top.add_entry("bonds", "BB")  # too few atoms
        with pytest.raises(ValueError, match="not found"):
            simple_top.add_entry("bonds", "BB", "GHOST")  # unknown atom
    
    def test_add_entry_params_stored(self, simple_top):
        simple_top.add_entry("bonds", "BB", "SC2", params=["1", "0.47", "3800"])
        assert simple_top.bonds[-1]["params"] == ["1", "0.47", "3800"]
    
    def test_remove_atom_raises(self, simple_top):
        with pytest.raises(ValueError, match="not found"):
            simple_top.remove_atom("GHOST")
    
    def test_remove_atom_cascades_all_sections(self, vs_top):
        vs_top.remove_atom("SC1")
        for bond in vs_top.bonds:
            assert "SC1" not in (bond["i"], bond["j"])
        for angle in vs_top.angles:
            assert "SC1" not in (angle["i"], angle["j"], angle["k"])
        for excl in vs_top.exclusions:
            assert "SC1" not in excl
        for vs in vs_top.virtual_sitesn:
            assert "SC1" not in vs["refs"] and vs["vs"] != "SC1"
        for vs in vs_top.virtual_sites2:
            assert "SC1" not in (vs["i"], vs["j"], vs["k"])
    
    def test_remove_entry_reversed(self, simple_top):
        simple_top.remove_entry("bonds", "SC1", "BB")  # reversed order
        assert len(simple_top.bonds) == 1
    
    def test_remove_entry_not_found_raises(self, simple_top):
        with pytest.raises(ValueError, match="not found"):
            simple_top.remove_entry("bonds", "BB", "SC2")  # doesn't exist
    
    def test_remove_entry_atoms_raises(self, simple_top):
        with pytest.raises(ValueError, match="remove_atom"):
            simple_top.remove_entry("atoms", "BB")

    def test_assign_fresh_indices_duplicate_raises(self, simple_top):
        """Directly injecting duplicate atom name should raise on rebuild."""
        simple_top.atoms.append({
            "nr": "99", "type": "P4", "resnr": "1", "residue": "MOL",
            "atom": "BB", "cgnr": "99", "charge": "0.0", "mass": None, "extra": []
        })
        with pytest.raises(ValueError, match="Duplicate"):
            simple_top._rebuild_sections()
    
    def test_add_atom_fallback_to_moleculetype(self):
        """With no atoms but a moleculetype, should inherit from moleculetype."""
        itp = """
            [ moleculetype ]
            MOL 1
            """
        top = ITPTopology(itp)
        top.add_atom("BB", "P4", "0.0")
        assert top.atoms[0]["residue"] == "MOL"
        assert top.atoms[0]["resnr"] == "1"
    
    def test_add_atom_fallback_to_default(self):
        """With no atoms and no moleculetype, should use MOL as default."""
        top = ITPTopology("")
        top.add_atom("BB", "P4", "0.0")
        assert top.atoms[0]["residue"] == "MOL"
        assert top.atoms[0]["resnr"] == "1"
    
    def test_remove_entry_virtual_sitesn(self, vs_top):
        vs_top.remove_entry("virtual_sitesn", "VS1", "BB", "SC1", "SC2")
        assert len(vs_top.virtual_sitesn) == 0
    
    def test_remove_entry_exclusion(self, simple_top):
        simple_top.remove_entry("exclusions", "BB", "SC1", "SC2")
        assert len(simple_top.exclusions) == 0
    
    def test_remove_entry_unknown_section_raises(self, simple_top):
        with pytest.raises(ValueError, match="Unknown"):
            simple_top.remove_entry("garbage", "BB", "SC1")