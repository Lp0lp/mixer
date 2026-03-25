import pytest
from mixer.parser import ITPTopology

def test_print_atoms(simple_top, capsys):
    simple_top.atoms.print()
    captured = capsys.readouterr()
    assert "BB" in captured.out
    assert "SC1" in captured.out
    assert "SC2" in captured.out
    assert "P4" in captured.out
    assert "SC4" in captured.out
    assert "0.0" in captured.out

def test_print_bonds(simple_top, capsys):
    simple_top.bonds.print()
    captured = capsys.readouterr()
    assert "BB" in captured.out
    assert "SC1" in captured.out
    assert "SC2" in captured.out

def test_print_angles(simple_top, capsys):
    simple_top.angles.print()
    captured = capsys.readouterr()
    assert "BB" in captured.out
    assert "SC1" in captured.out
    assert "SC2" in captured.out

def test_print_dihedrals(vs_top, capsys):
    vs_top.dihedrals.print()
    captured = capsys.readouterr()
    assert "BB" in captured.out
    assert "SC1" in captured.out

def test_print_virtual_sites2(vs_top, capsys):
    vs_top.virtual_sites2.print()
    captured = capsys.readouterr()
    assert "VS1" in captured.out
    assert "BB" in captured.out

def test_print_virtual_sitesn(vs_top, capsys):
    vs_top.virtual_sitesn.print()
    captured = capsys.readouterr()
    assert "VS1" in captured.out

def test_print_exclusions(simple_top, capsys):
    simple_top.exclusions.print()
    captured = capsys.readouterr()
    assert "BB" in captured.out
    assert "SC1" in captured.out

def test_print_empty_section(simple_top, capsys):
    simple_top.dihedrals.print()
    captured = capsys.readouterr()
    assert "(empty)" in captured.out

def test_print_constraints(simple_top, capsys):
    simple_top.constraints.print()
    captured = capsys.readouterr()
    assert "BB" in captured.out
    assert "SC1" in captured.out

def test_print_pairs(simple_top, capsys):
    simple_top.pairs.print()
    captured = capsys.readouterr()
    assert "BB" in captured.out
    assert "SC2" in captured.out