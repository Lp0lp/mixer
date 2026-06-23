# Mixer

Mixer is a Python toolkit for parsing, editing, merging, and assembling [Martini 3](http://cgmartini.nl/) coarse-grained topologies. It lets you programmatically manipulate and merge topology fragments and generate relaxed starting structures ready for GROMACS simulations. While built to be general, Mixer has a focus on lipids — it can generate Martini 3 `.itp` files for arbitrary acyl chains that can then be merged into complete molecules.

---
 
## Features
 

- **Topology parser & writer** — read, edit, merge, and re-index GROMACS `.itp` files with atom-name-based connectivity (no manual index tracking); supports merging overlapping fragments into a complete molecule with all bonded parameters resolved
- **Structure generator** — build a randomised starting structure, solvate it, energy-minimise with GROMACS, and extract a clean PDB
- **Acyl chain builder** — generate Martini 3 `.itp` files for arbitrary acyl chains from a compact shorthand notation
- **INSANE entry writer** — convert a relaxed structure into an entry for the INSANE membrane builder
 
---
 
## Installation
 
Clone the repository:
```bash
git clone https://github.com/Lp0lp/mixer.git
```
Create a virtual environment with the dependencies (e.g. with **conda**):
```bash
cd mixer
conda env create -f environment.yml
```
This will create a conda environment with the name mixer.

Then activate your environment:
```bash
conda activate mixer
```

Install Mixer with pip:
```bash
pip install .
```
---
 
## Fragment-based assembly of molecules
 
A complete molecule topology can be assembled from a set of overlapping fragments, where each fragment covers one region of the molecule. Fragments are allowed to share atoms: when two fragments define the same atom name, the atom is deduplicated in the merge, but all bonded parameters from **both** fragments are retained.
 
This is what makes cross-boundary interactions work. A bond, angle, or dihedral that spans two fragments — say, the angle between the last bead of one region and the first bead of the next — can be encoded in either fragment, or in a dedicated bridge fragment overlaid on both neighbours. You do not need to write a monolithic topology by hand; you only need to describe each region once with sufficient overlap at the boundaries, and the merge assembles the complete parameter set automatically.
 
For lipids, this means a headgroup, a linker, and any number of tails can be developed and validated independently, then combined:
 
```
  headgroup.itp        tail_A.itp
  ┌──────────────┐     ┌──────────────┐
  │ NC3 PO4 GL1  │     │ GL1 C1A C2A  │
  │ bonds/angles │     │ bonds/angles │
  └──────┬───────┘     └───┬──────────┘
         │   shared atom   │
         └────── GL1 ──────┘
                  │
            merged topology
         ┌────────────────────┐
         │ NC3 PO4 GL1 C1A C2A│
         │ all bonds + angles │
         └────────────────────┘
```
 
**Fragments must be designed with this workflow in mind.** Two requirements apply:
 
- **Atom names must be unique within the final molecule** and **consistent across fragments** — the merge matches atoms by name, so a shared atom must carry the same name in every fragment that references it.
- **Overlap must be sufficient to parameterise all boundary interactions** — if an angle spans three atoms across a fragment boundary, at least one fragment must contain all three of those atoms so the angle term is captured.
 
The first fragment passed to `merge_itps` is the base. Each subsequent fragment is merged into it, and the `redundancy_priority` list controls what happens when two fragments define the same bonded entry: by default the base wins, but you can flip this per-fragment to let a later fragment override.
 
---