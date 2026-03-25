import subprocess
from collections import Counter
from importlib.resources import files
from pathlib import Path
import warnings

import MDAnalysis as md
import numpy as np

from .parser import ITPTopology

warnings.filterwarnings("ignore", message="missing dimension")
warnings.filterwarnings("ignore", message="Empty box")
warnings.filterwarnings("ignore", message="Element information is missing")
warnings.filterwarnings("ignore", message="Failed to guess the mass")
'''
Structure manipulators live here.
'''

def generate_relaxed_structure(filename,
                FFitp=None, FFbonded=None, SolvITP=None,
                Waterbox=None, MinMDP=None, RelMDP=None,
                nt=5, cylinder_radius=0.3, cylinder_k=250,
                relax=True, box_size=None, cleanup=True):
    """
    Generate a relaxed coarse-grained structure from a Martini `.itp` file.

    All intermediate and output files are written to the current working
    directory. The final output is ``structure.pdb``.

    Parameters
    ----------
    filename : str or Path
        Path to the molecule `.itp` file.
    FFitp, FFbonded, SolvITP : str or Path, optional
        Martini force-field `.itp` files. Package-bundled defaults used when None.
    Waterbox : str or Path, optional
        Solvent `.gro` box. Package default used when None.
    MinMDP, RelMDP : str or Path, optional
        GROMACS `.mdp` files for minimisation and relaxation.
    nt : int
        Number of threads for GROMACS mdrun (default 5).
    cylinder_radius : float
        Radius (nm) of the flat-bottomed cylindrical position restraint (default 0.3).
    cylinder_k : float
        Force constant (kJ/mol/nm²) of the cylindrical restraint (default 250).
    relax : bool
        Run a short relaxation MD after minimisation (default True).
        Set to False to skip relaxation and use the minimised structure directly.
    box_size : float, optional
        Edge length (nm) of the dodecahedral box for ``gmx editconf``.
        If None, inferred from atom count using the same sphere-packing logic
        as ``_create_random_structure_from_file``, plus a 4 nm solvation buffer.
    cleanup : bool
        Remove intermediate GROMACS files after a successful run.

    Returns
    -------
    Path
        Absolute path to the final ``structure.pdb``.
    """
    if FFitp    is None: FFitp    = files("mixer.data.itps") / "martini_v3.0.0.itp"
    if FFbonded is None: FFbonded = files("mixer.data.itps") / "martini_v3.0.0_ffbonded_v2.itp"
    if SolvITP  is None: SolvITP  = files("mixer.data.itps") / "martini_v3.0.0_solvents_v1.itp"
    if Waterbox is None: Waterbox = files("mixer.data.mdps") / "water.gro"
    if MinMDP   is None: MinMDP   = files("mixer.data.mdps") / "min.mdp"
    if RelMDP   is None: RelMDP   = files("mixer.data.mdps") / "rel.mdp"

    filename = Path(filename).resolve()

    # build initial random structure
    _create_random_structure_from_file(filename, outpath="initial.gro")

    u        = md.Universe("initial.gro")
    counts   = Counter(u.atoms.residues.resnames)
    num_atom = len(u.atoms)

    # topology file
    topol_lines = [
        f'#include "{FFitp}"\n',
        f'#include "{FFbonded}"\n',
        f'#include "{SolvITP}"\n',
        f'#include "{filename}"\n',
        '#include "posres.itp"\n',
        '[system]\n',
        f'Structure Fixer: {filename.name}\n',
        '\n',
        '[ molecules ]\n',]

    for resname, count in counts.items():
        topol_lines.append(f"{resname}    {count}\n")
    Path("topol.top").write_text("".join(topol_lines))

    # Define Position restraints
    # Flat-bottomed cylinder along Z (type 2), g=3 → Z-axis
    posres_lines = [
        "; Position restraint file for GROMACS\n",
        "; Flat-bottomed cylinder of radius 0.3 nm, axis along Z\n",
        "[ position_restraints ]\n",
        ";  atom  funct  g  r(nm)   k\n",]

    for idx in range(1, num_atom + 1):
        posres_lines.append(f"  {idx}  2  3  {cylinder_radius}  {cylinder_k}\n")
    Path("posres.itp").write_text("".join(posres_lines))

    # GROMACS Relaxation
    with open("structure_generator.log", "w") as log:

        if box_size is None:
            bead_volume = (4.0 / 3.0) * np.pi * (2.5 ** 3)  # sphere r=2.5 Å
            box_size    = round((num_atom * bead_volume) ** (1.0 / 3.0) / 10.0 + 4.0, 1)  # Å → nm + buffer

        _run(["gmx", "editconf", "-f", "initial.gro",
              "-o", "box.gro", "-box", str(box_size), str(box_size), str(box_size),
              "-bt", "dodecahedron"], log=log)

        # Shift atoms off-centre for cylindrical restraints
        u = md.Universe("box.gro")
        pos = u.atoms.positions.copy()
        pos[:, 0] = 40.0   # 40 Å = 4 nm
        pos[:, 1] = 40.0
        u.atoms.positions = pos
        u.atoms.write("restraints.gro")

        _run(["gmx", "solvate", "-cp", "box.gro",
              "-cs", str(Waterbox), "-o", "watered.gro",
              "-p", "topol.top"], log=log)

        # Minimisation
        _run(["gmx", "grompp", "-f", str(MinMDP),
              "-c", "watered.gro", "-p", "topol.top",
              "-o", "min.tpr", "-maxwarn", "5",
              "-r", "restraints.gro"], log=log)
        _run(["gmx", "mdrun", "-deffnm", "min",
              "-v", "-nt", str(nt), "-pin", "on", "-pinoffset", "0"], log=log)

        # Relaxation
        if relax:
            _run(["gmx", "grompp", "-f", str(RelMDP),
                  "-c", "min.gro", "-p", "topol.top",
                  "-o", "rel.tpr", "-maxwarn", "5",
                  "-r", "restraints.gro"], log=log)
            _run(["gmx", "mdrun", "-deffnm", "rel",
                  "-v", "-nt", str(nt), "-pin", "on", "-pinoffset", "0"], log=log)
            final_gro, final_tpr = "rel.gro", "rel.tpr"
        else:
            final_gro, final_tpr = "min.gro", "min.tpr"

        # write clean PDB
        u = md.Universe(final_gro)
        u.select_atoms("not resname W").write("index.ndx", mode="w", name="Clean")

        _run(["gmx", "trjconv", "-f", final_gro, "-s", final_tpr,
              "-pbc", "whole", "-o", "structure.pdb",
              "-conect", "-n", "index.ndx"],
             log=log, input_text="Clean\n")

        # Strip ENDMDL
        pdb = Path("structure.pdb")
        pdb.write_text("".join(
            l for l in pdb.read_text().splitlines(keepends=True)
            if "ENDMDL" not in l))

    if cleanup:
        _cleanup_temp_files(Path.cwd())

    return Path("structure.pdb").resolve()

def structure2insane(structure, write=True,
                     output='./insane_entry.txt',
                     moltype="lipid", anchor=None):
    
    """
    Convert a relaxed structure file into an INSANE lipid entry.
 
    The anchor bead is used as the z origin and is assumed to be the tail
    bead deepest in the membrane. z coordinates are always positive.
 
    Parameters
    ----------
    structure : str or Path
        Path to the structure file (PDB or GRO).
    write : bool
        Write the entry to *output* when ``True``.
    output : str or Path
        Destination file for the INSANE entry.
    moltype : str
        Molecule-type label used inside the INSANE entry (default ``"lipid"``).
    anchor : str, optional
        Atom name to use as the z origin, i.e. the closest to membrane core.
        Defaults to the last atom in the structure.
 
    Returns
    -------
    list[str]
        Lines of the INSANE entry.
    """
    
    u = md.Universe(str(structure))
    resname = u.atoms.residues.resnames[0]
    names   = u.atoms.names
 
    # Resolve anchor atom
    if anchor is not None:
        sel = u.select_atoms(f"name {anchor}")
        if len(sel) == 0:
            raise ValueError(f"Anchor atom '{anchor}' not found in structure.")
        anchor_pos = sel.positions[0]
    else:
        anchor_pos = u.atoms.positions[-1]
 
    # Centre on origin; convert Å → nm
    pos_nm     = (u.atoms.positions - u.atoms.center_of_geometry()) / 10.0
    anchor_z   = (anchor_pos - u.atoms.center_of_geometry())[2] / 10.0
 
    x = pos_nm[:, 0]
    y = pos_nm[:, 1]
    z = np.abs(pos_nm[:, 2] - anchor_z)
 
    # Trailing comma ensures single-bead molecules are valid Python tuples
    x_str    = ", ".join(f"{v:.1f}" for v in x) + ","
    y_str    = ", ".join(f"{v:.1f}" for v in y) + ","
    z_str    = ", ".join(f"{v:.1f}" for v in z) + ","
    name_str = " ".join(names)
 
    lines = [f'moltype = "{moltype}"\n',
             f"lipidsx[moltype] = ({x_str})\n",
             f"lipidsy[moltype] = ({y_str})\n",
             f"lipidsz[moltype] = ({z_str})\n",
             "lipidsa.update({\n",
             f'    "{resname}": (moltype, "{name_str}"),\n',
             "})\n",]
 
    if write:
        Path(output).write_text("".join(lines))
    else:
        return lines

def _run(cmd, *, log=None, env=None, input_text=None, cwd=None):
    '''
    Short helper to assist when using subprocess to run gmx.
    '''
    subprocess.run(cmd, input=input_text,
        text=input_text is not None,
        stdout=log, stderr=subprocess.STDOUT, #if log is None goes to term.
        env=env, cwd=cwd, check=True,)

def _parse_file(filename):
    """Return a list of ``{atom_name, resname, resid}`` dicts from an `.itp` [ atoms ] section.
 
    Delegates to ITPTopology for robust parsing.
    """
 
    top = ITPTopology.from_file(filename)
 
    return [{"atom_name":atom["atom"], "resname":atom["residue"], "resid":int(atom["resnr"])}
            for atom in top.atoms]
 

def _create_random_structure_from_file(filename, outpath='./initial.gro', spacing=5.0):
    """Read an `.itp` file and write a structure file with randomised atom positions.
 
    The cubic box edge is scaled so the mean bead-bead spacing equals
    *spacing* Å, keeping the starting geometry sensible across molecule
    sizes (a ~12-bead Martini lipid at the default spacing gives ~26 Å).
 
    Parameters
    ----------
    filename : str or Path
        Molecule `.itp` file to read atom names from.
    outpath : str or Path
        Destination `.gro` (or `.pdb`) file.
    spacing : float
        Target mean bead spacing in Å (default 5.0).
 
    Returns
    -------
    Path
        Absolute path to the written structure file.
    """
    outpath    = Path(outpath).resolve()
    atoms_info = _parse_file(filename)
    n_atoms    = len(atoms_info)
 
    sol = md.Universe.empty(
        n_atoms,
        n_residues=1,
        atom_resindex=[0] * n_atoms,
        residue_segindex=[0],
        trajectory=True,
    )
    sol.add_TopologyAttr('name',    [info['atom_name'] for info in atoms_info])
    sol.add_TopologyAttr('resname', [atoms_info[0]['resname']])
    sol.add_TopologyAttr('resid',   [atoms_info[0]['resid']])
 
    # Volume per bead approximated as a sphere of diameter `spacing`,
    # rather than a cube — gives a ~48 % smaller box edge.
    bead_volume = (4.0 / 3.0) * np.pi * (spacing / 2.0) ** 3
    box_edge = (n_atoms * bead_volume) ** (1.0 / 3.0)  # Å
    sol.atoms.positions = np.random.default_rng().uniform(0.0, box_edge, size=(n_atoms, 3))
 
    sol.atoms.write(str(outpath))
    return outpath
    

def _cleanup_temp_files(workdir):
    """Remove intermediate GROMACS files from *workdir*."""
    workdir = Path(workdir)
 
    patterns = [
        "initial.gro", "box.gro", "watered.gro",
        "restraints.gro", "posres.itp", "topol.top",
        "mdout.mdp", "index.ndx",
        "min.*", "rel.*", "*.cpt",
        "#*#",      # GROMACS backup files: #filename.ext.N#
    ]
 
    for pattern in patterns:
        for f in workdir.glob(pattern):
            try:
                f.unlink()
            except (PermissionError, FileNotFoundError):
                pass
 