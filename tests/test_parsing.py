from mixer.parser import ITPTopology
import pytest

class TestParsing:
    def test_atom_count(self, simple_top):
        assert len(simple_top.atoms) == 3

    def test_atom_fields(self, simple_top):
        bb = simple_top.atoms[0]
        assert bb["atom"]    == "BB"
        assert bb["type"]    == "P4"
        assert bb["charge"]  == "0.0"
        assert bb["residue"] == "MOL"
        assert bb["resnr"]   == "1"
        assert bb["mass"]    is None      
        assert bb["extra"]   == []

    def test_bonds_use_names_not_indices(self, simple_top):
        assert simple_top.bonds[0]["i"] == "BB"
        assert simple_top.bonds[0]["j"] == "SC1"

    def test_moleculetype(self, simple_top):
        assert simple_top.moleculetype["name"] == "MOL"
        assert simple_top.moleculetype["nrexcl"] == "1"

    def test_bond_fields(self, simple_top):
        bond = simple_top.bonds[0]
        assert bond["i"]      == "BB"
        assert bond["j"]      == "SC1"
        assert bond["params"] == ["1", "0.47", "3800"]

    def test_virtual_sitesn_fields(self, vs_top):
        vs = vs_top.virtual_sitesn[0]
        assert vs["vs"]   == "VS1"
        assert vs["type"] == "1"
        assert vs["refs"] == ["BB", "SC1", "SC2"]
        
    def test_virtual_sites2_fields(self, vs_top):
        vs = vs_top.virtual_sites2[0]
        assert vs["i"]      == "VS1"   # vs atom
        assert vs["j"]      == "BB"    # ref 1
        assert vs["k"]      == "SC1"   # ref 2
        assert vs["params"] == ["1", "0.5"]
    
    def test_exclusions_fields(self, simple_top):
        excl = simple_top.exclusions[0]
        assert excl == ["BB", "SC1", "SC2"]
    
    def test_duplicate_atom_name_raises(self):
        itp = """
             [ moleculetype ]
             MOL 1
             [ atoms ]
             1 P4 1 MOL BB 1 0.0
             2 P4 1 MOL BB 2 0.0
             """
        with pytest.raises(ValueError, match="Duplicate"):
            ITPTopology(itp)

    def test_malformed_atom_raises(self):
        itp = """
             [ moleculetype ]
             MOL 1
             [ atoms ]
             1 P4 1
             """
        with pytest.raises(ValueError, match="Malformed"):
            ITPTopology(itp)

    def test_unknown_atom_index_raises(self):
        itp = """
             [ moleculetype ]
             MOL 1
             [ atoms ]
             1 P4 1 MOL BB 1 0.0
             [ bonds ]
             1 99 1 0.47 3800
             """
        with pytest.raises(ValueError, match="not found"):
            ITPTopology(itp)

    def test_ifdef_warns(self):
        itp = """
             [ moleculetype ]
             MOL 1
             [ atoms ]
             1 P4 1 MOL BB 1 0.0
             #ifdef SOMETHING
             [ bonds ]
             1 1 1
             #endif
             """
        with pytest.warns(UserWarning, match="Preprocessor"):
            ITPTopology(itp)

    def test_empty_topology(self):
        top = ITPTopology("")
        assert len(top.atoms) == 0
        assert len(top.bonds) == 0

    def test_maps_are_inverses(self, simple_top):
        name = "SC2"
        nr = simple_top.name_to_nr[name]
        assert simple_top.nr_to_name[nr] == name


    def test_virtual_sitesn_unknown_atom_raises(self):
        itp = """
            [ moleculetype ]
            MOL 1
            [ atoms ]
            1 P4 1 MOL BB 1 0.0
            [ virtual_sitesn ]
            1 1 99 2
            """
        with pytest.raises(ValueError, match="not found"):
            ITPTopology(itp)
    
    def test_exclusions_unknown_atom_raises(self):
        itp = """
            [ moleculetype ]
            MOL 1
            [ atoms ]
            1 P4 1 MOL BB 1 0.0
            [ exclusions ]
            1 99
            """
        with pytest.raises(ValueError, match="not found"):
            ITPTopology(itp)
    
    def test_moleculetype_extra_fields(self):
        itp = """
            [ moleculetype ]
            MOL 1 extra_field
            [ atoms ]
            1 P4 1 MOL BB 1 0.0
            """
        top = ITPTopology(itp)
        assert top.moleculetype["extra"] == ["extra_field"]