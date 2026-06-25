"""
ITPTopology — GROMACS/Martini .itp topology parser and writer.

Overview
--------
Reads a .itp file with numeric atom indices, stores all connectivity
internally using atom names, and writes back out with fresh sequential
indices. This means topologies can be safely merged, reordered, and
modified without tracking index offsets manually.

Workflow
---------
1. _parse_raw_sections   — splits raw text into named section buckets
2. _build_internal_model — converts section lines into structured dicts
                           (atoms, bonds, angles, etc.), keyed by atom name
3. _rebuild_sections     — reassigns fresh 1-based indices and serialises
                           back to text; called automatically before writing

Structured attributes
---------------------
After parsing, connectivity is accessible as TopologySection lists:

    top.atoms           # list of atom dicts
    top.bonds           # list of {i, j, params}
    top.angles          # list of {i, j, k, params}
    top.dihedrals       # list of {i, j, k, l, params}
    top.constraints     # list of {i, j, params}
    top.pairs           # list of {i, j, params}
    top.virtual_sites1  # list of {i, j, params}         (vs + 1 ref)
    top.virtual_sites2  # list of {i, j, k, params}      (vs + 2 refs)
    top.virtual_sites3  # list of {i, j, k, l, params}   (vs + 3 refs)
    top.virtual_sites4  # list of {i, j, k, l, m, params}(vs + 4 refs)
    top.virtual_sitesn  # list of {vs, type, refs}        (variable refs)
    top.exclusions      # list of [name, name, ...]
    top.moleculetype    # dict {name, nrexcl}
    top.other_sections  # dict of unrecognised raw sections

All atom references in the above use atom names, not numeric indices.
Atom names must be unique within the molecule.

Public API
----------
    ITPTopology.from_file(filename)          load from file path
    top.merge(other, ...)                    merge another topology in-place
    top.set_resname(resname)                 rename molecule and all residues
    top.write_itp(filename)                  write to .itp file
    top.atoms.print()                        pretty-print any section

Limitations
-----------
- Atom names must be unique (enforced on parse and merge).
- Preprocessor directives (#ifdef, #define, etc.) are not supported;
  a UserWarning is raised and the directive is skipped.

Example
-------
    top1 = ITPTopology.from_file("molecule_a.itp")
    top2 = ITPTopology.from_file("molecule_b.itp")
    top1.merge(top2, remove_redundant=True, keep_second=False)
    top1.set_resname("MERGED")
    top1.write_itp("merged.itp")
"""

import re
from collections import defaultdict
from copy import deepcopy
import warnings

class TopologySection(list):
    """
    Small container for topology sections (atoms, bonds, angles, etc.)
    that behaves like a list but adds nice printing helpers.
    """

    def __init__(self, data=None, kind="generic"):
        super().__init__(data or [])
        self.kind = kind

    def __repr__(self):
        return f"<TopologySection {self.kind} ({len(self)} entries)>"

    def print(self):
        print(f"[{self.kind}]")

        if not self:
            print("(empty)")
            return

        if self.kind == "atoms":
            print(f"{'nr':>4} {'atom':>6} {'type':>8} {'resnr':>6} {'res':>6} {'q':>6} {'m':>6}")
            print("-" * 54)
            for a in self:
                mass = a["mass"] if a["mass"] is not None else ""
                print(f"{a['nr']:>4} {a['atom']:>6} {a['type']:>8} "
                      f"{a['resnr']:>6} {a['residue']:>6} {a['charge']:>6} {mass:>6}")

        elif self.kind in ("bonds", "constraints", "pairs"):
            print(f"{'i':>6} {'j':>6}  params")
            print("-" * 40)
            for x in self:
                print(f"{x['i']:>6} {x['j']:>6}  {' '.join(x['params'])}")

        elif self.kind == "angles":
            print(f"{'i':>6} {'j':>6} {'k':>6}  params")
            print("-" * 48)
            for x in self:
                print(f"{x['i']:>6} {x['j']:>6} {x['k']:>6}  {' '.join(x['params'])}")

        elif self.kind == "dihedrals":
            print(f"{'i':>6} {'j':>6} {'k':>6} {'l':>6}  params")
            print("-" * 56)
            for x in self:
                print(f"{x['i']:>6} {x['j']:>6} {x['k']:>6} {x['l']:>6}  "
                      f"{' '.join(x['params'])}")

        elif self.kind in ("virtual_sites1", "virtual_sites2", "virtual_sites3", "virtual_sites4"):
            n = {"virtual_sites1": 2, "virtual_sites2": 3,
                 "virtual_sites3": 4, "virtual_sites4": 5}[self.kind]
            labels = [chr(ord('i') + idx) for idx in range(n)]
            print("  ".join(f"{l:>6}" for l in labels) + "  params")
            print("-" * (8 * n))
            for x in self:
                print("  ".join(f"{x[l]:>6}" for l in labels) + f"  {' '.join(x['params'])}")
        
        elif self.kind == "virtual_sitesn":
            print(f"{'vs':>6} {'type':>6}  refs")
            print("-" * 40)
            for x in self:
                if x["type"] == "3":
                    ref_str = "  ".join(f"{r['ref']} {r['weight']}" for r in x["refs"])
                else:
                    ref_str = " ".join(x["refs"])
                print(f"{x['vs']:>6} {x['type']:>6}  {ref_str}")

        elif self.kind == "exclusions":
            print("excluded atoms")
            print("-" * 40)
            for x in self:
                print(" ".join(x))
                
        else:
            for item in self:
                print(item)

        print()

                
class ITPTopology:
    """
    Parser and writer for GROMACS/Martini .itp topology files.

    Reads numeric atom indices from file, stores connectivity internally
    using atom names, and writes back out with fresh sequential indices.

    Atom names must be unique within a molecule.

    Parameters
    ----------
    file_content : str
        Raw string content of an .itp file.

    Examples
    --------
    >>> top = ITPTopology.from_file("molecule.itp")
    >>> top.atoms.print()
    >>> top.merge(other_top)
    >>> top.write_itp("merged.itp")
    """

    # Constants
    # ------------------------------------------------------------------
    
    STRUCTURED_SECTIONS = [
            "moleculetype",
            "atoms",
            "bonds",
            "constraints",
            "angles",
            "dihedrals",
            "virtual_sitesn",
            "virtual_sites1",
            "virtual_sites2",
            "virtual_sites3",
            "virtual_sites4",
            "pairs",
            "exclusions",]
    
    # maps section name -> number of atom indices on each line
    NBODY_SECTIONS = {
        "bonds": 2,
        "constraints": 2,
        "pairs": 2,
        "angles": 3,
        "dihedrals": 4,
        "virtual_sites1": 2,  # vs + 1 ref
        "virtual_sites2": 3,  # vs + 2 refs
        "virtual_sites3": 4,  # vs + 3 refs
        "virtual_sites4": 5,} # vs + 4 refs

    NBODY_COMMENTS = {"bonds":         ";  i     j     params",
                      "constraints":   ";  i     j     params",
                      "pairs":         ";  i     j     params",
                      "angles":        ";  i     j     k     params",
                      "dihedrals":     ";  i     j     k     l     params",
                      "virtual_sites1":";  vs    ref   params",
                      "virtual_sites2":";  vs    i     j     params",
                      "virtual_sites3":";  vs    i     j     k     params",
                      "virtual_sites4":";  vs    i     j     k     l   params",}

    
    # Construction
    # ------------------------------------------------------------------
    
    def __init__(self, file_content):
        self.file_content = file_content

        # Raw parsed text sections from input.
        self.sections = defaultdict(list)

        # Structured internal representation.
        self.moleculetype = None
        self.atoms = []
        for section in self.NBODY_SECTIONS:
            setattr(self, section, [])
        self.virtual_sitesn = []
        self.exclusions = []
        self.other_sections = {}
        self.nr_to_name = {}
        self.name_to_nr = {}
        self._parse_raw_sections()
        self._build_internal_model()
        self._rebuild_sections()

    @classmethod
    def from_file(cls, filename):
        """Load an ITPTopology directly from a file path."""
        with open(filename, "r") as f:
            file_content = f.read()
        return cls(file_content)

    # Parsing (Tidy the raw .itp data into nicely organized stuffs)
    # ------------------------------------------------------------------
    
    def _parse_raw_sections(self):
        """
        Split raw file content into named sections, storing lines per section.
        Warns and skips preprocessor directives (#ifdef, #define, etc.).
        """
        
        current_section = None
        section_pattern = re.compile(r"^\[\s*([^\]]+?)\s*\]")
        ifdef_pattern = re.compile(r"^#\s*(ifdef|ifndef|endif|else|define|include)")
    
        for line in self.file_content.splitlines():
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if ifdef_pattern.match(line):
                warnings.warn(f"Preprocessor directive detected: {line!r}. "
                               "Directives are not supported and will be ignored. "
                               "The parsed topology may be incomplete or incorrect.",
                               UserWarning, stacklevel=2)
                continue
    
            section_match = section_pattern.match(line)
            if section_match:
                current_section = section_match.group(1).strip()
                continue
    
            if current_section:
                self.sections[current_section].append(line)

    def _build_internal_model(self):
        """Parse raw section lines into structured Python objects."""
        self.moleculetype = self._parse_moleculetype_section(
            self.sections.get("moleculetype", []))
        self.atoms = self._parse_atoms_section(
            self.sections.get("atoms", []))
        self._refresh_maps()
        for section, n in self.NBODY_SECTIONS.items():
            setattr(self, section, self._parse_nbody_section(self.sections.get(section, []), n))        
        self.virtual_sitesn = self._parse_virtual_sitesn_section(
            self.sections.get("virtual_sitesn", []))
        self.exclusions = self._parse_exclusions_section(
            self.sections.get("exclusions", []))
        
        self.other_sections = {}
        for section, lines in self.sections.items():
            if section not in set(self.STRUCTURED_SECTIONS):
                self.other_sections[section] = list(lines)

    def _refresh_maps(self):
        """
        Does the index to atom name matching.
        Rebuilds nr_to_name and name_to_nr from current atom list.
        Raises ValueError on duplicate atom names.
        """
        
        self.nr_to_name = {}
        self.name_to_nr = {}

        for atom in self.atoms:
            nr = atom["nr"]
            name = atom["atom"]

            self.nr_to_name[nr] = name
            if name in self.name_to_nr:
                raise ValueError(f"Duplicate atom name '{name}' found in [ atoms ]. "
                                  "Atom names must be unique! Sorry!")
            self.name_to_nr[name] = nr

    def _parse_moleculetype_section(self, lines):
        """Parse [ moleculetype ] into a dict with name, nrexcl, and any extra fields."""
        for line in lines:
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                return {"name": parts[0],
                        "nrexcl": parts[1],
                        "extra": parts[2:] if len(parts) > 2 else []}
        return None
    
    def _parse_atoms_section(self, lines):
        """Parse [ atoms ] into a list of atom dicts. Raises ValueError on weird lines."""
        atoms = []
        for line in lines:
            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Malformed [ atoms ] line: {line}")

            atom = {"nr": str(parts[0]),
                    "type": parts[1],
                    "resnr": parts[2],
                    "residue": parts[3],
                    "atom": parts[4],
                    "cgnr": parts[5],
                    "charge": parts[6],
                    "mass": parts[7] if len(parts) > 7 else None,
                    "extra": parts[8:] if len(parts) > 8 else []}
            atoms.append(atom)
        return atoms

    def _parse_nbody_section(self, lines, n):
        """
        Parse a bonded interaction section with n atom indices per line.
        Atoms are stored by name. Raises ValueError on unknown atom indices.
        """
        
        entries = []
        labels = tuple(chr(ord('i') + idx) for idx in range(n))
        for line in lines:
            parts = line.split()
            if len(parts) < n:
                continue
            try:
                atoms = [self.nr_to_name[x] for x in parts[:n]]
            except KeyError as e:
                raise ValueError(f"Atom index {e} not found in [ atoms ] — "
                                 f"referenced in line: {line!r}") from None
            entry = dict(zip(labels, atoms))
            entry["params"] = parts[n:]
            entries.append(entry)
        return entries

    def _parse_virtual_sitesn_section(self, lines):
        """
        Parse [ virtual_sitesn ] lines.

        Type 3 uses an interleaved idx/weight format:
            vs  3  ref_idx  weight  ref_idx  weight  ...
        refs are stored as [{"ref": name, "weight": "0.28005"}, ...]

        All other types use plain atom index lists:
            vs  type  ref_idx  ref_idx  ...
        refs are stored as [name, name, ...]
        """
        entries = []
        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                vs = self.nr_to_name[parts[0]]
                site_type = parts[1]
                rest = parts[2:]

                if site_type == "3":
                    if len(rest) % 2 != 0:
                        raise ValueError(
                            f"Type 3 virtual_sitesn expects interleaved idx/weight pairs, "
                            f"but got an odd number of remaining tokens in line: {line!r}")
                    refs = []
                    for i in range(0, len(rest), 2):
                        refs.append({"ref": self.nr_to_name[rest[i]], "weight": rest[i + 1]})
                else:
                    refs = [self.nr_to_name[x] for x in rest]

            except KeyError as e:
                raise ValueError(f"Atom index {e} not found in [ atoms ] — "
                                 f"referenced in line: {line!r}") from None
            entries.append({"vs": vs, "type": site_type, "refs": refs})
        return entries
    
    def _parse_exclusions_section(self, lines):
        entries = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                parts = [self.nr_to_name[x] for x in parts]
            except KeyError as e:
                raise ValueError(f"Atom index {e} not found in [ atoms ] — "
                                 f"referenced in line: {line!r}") from None
            entries.append(parts)
        return entries

        
    # Formatting (structured data -> text)
    # ------------------------------------------------------------------
    
    def _assign_fresh_indices(self):
        """
        Renumber atoms sequentially (1-based) in current order.
        Updates both nr and cgnr. Raises ValueError on duplicate names.
        """
        new_name_to_nr = {}
        for idx, atom in enumerate(self.atoms, start=1):
            name = atom["atom"]
            if name in new_name_to_nr:
                raise ValueError(f"Duplicate atom name '{name}' present after merge.")
    
            atom["nr"] = str(idx)
            atom["cgnr"] = str(idx)
            new_name_to_nr[name] = str(idx)
    
        self.name_to_nr = new_name_to_nr
        self.nr_to_name = {v: k for k, v in new_name_to_nr.items()}

    def _format_moleculetype_line(self):
        if self.moleculetype is None:
            return None
    
        extra = " ".join(self.moleculetype.get("extra", []))
        return (f"{self.moleculetype['name']:<8} "
                f"{self.moleculetype['nrexcl']:>5} "
                f"{extra}").rstrip()
    
    def _format_atom_line(self, atom):
        mass = atom["mass"] if atom["mass"] is not None else ""
        extra = " ".join(atom.get("extra", []))
    
        return (f"{atom['nr']:>5} "
                f"{atom['type']:<8} "
                f"{atom['resnr']:>5} "
                f"{atom['residue']:<6} "
                f"{atom['atom']:<6} "
                f"{atom['cgnr']:>5} "
                f"{atom['charge']:>8} "
                f"{mass:>6} "
                f"{extra}").rstrip()

    def _format_nbody_line(self, entry, n):
        """Format a bonded entry back to a whitespace-aligned index string."""
        labels = tuple(chr(ord("i") + idx) for idx in range(n))
        idxs = [self.name_to_nr[entry[label]] for label in labels]
        return " ".join(f"{x:>5}" for x in idxs) + "  " + " ".join(entry["params"])

    def _format_virtual_sitesn_line(self, entry):
        vs = self.name_to_nr[entry["vs"]]
        if entry["type"] == "3":
            # interleaved: ref_idx weight ref_idx weight ...
            ref_parts = []
            for r in entry["refs"]:
                ref_parts.append(self.name_to_nr[r["ref"]])
                ref_parts.append(r["weight"])
            return " ".join([vs, entry["type"], *ref_parts])
        else:
            refs = [self.name_to_nr[x] for x in entry["refs"]]
            return " ".join([vs, entry["type"], *refs])

    def _format_exclusions_line(self, entry):
        return " ".join(self.name_to_nr[x] for x in entry)

    def _rebuild_sections(self):
        """
        Regenerate self.sections from structured data.
        Assigns fresh indices, formats all sections, then wraps in TopologySection.
        Called automatically after merge(), set_resname(), and write_itp().
        """
        self._assign_fresh_indices()
        sections = {}

        if self.moleculetype:
            sections["moleculetype"] = [";  name      nrexcl", self._format_moleculetype_line()]

        if self.atoms:
            sections["atoms"] = [";  nr  type      resnr residue atom   cgnr   charge   mass"]
            sections["atoms"].extend(self._format_atom_line(atom) for atom in self.atoms)
        
        for section, n in self.NBODY_SECTIONS.items():
            entries = getattr(self, section)
            if entries:
                sections[section] = [self.NBODY_COMMENTS[section]]
                sections[section].extend(self._format_nbody_line(x, n) for x in entries)
        
        if self.virtual_sitesn:
            sections["virtual_sitesn"] = [";  vs    type   refs"]
            sections["virtual_sitesn"].extend(self._format_virtual_sitesn_line(x) for x in self.virtual_sitesn)
        
        if self.exclusions:
            sections["exclusions"] = [";  excluded atoms"]
            sections["exclusions"].extend(self._format_exclusions_line(x) for x in self.exclusions)

        for section, lines in self.other_sections.items():
            sections[section] = list(lines)

        self.sections = sections
        self._wrap_sections()

    def _wrap_sections(self):
        """Wrap structured lists in TopologySection for pretty printing."""
        ## skips moleculetype
        for section in self.STRUCTURED_SECTIONS[1:]:
            setattr(self, section, TopologySection(getattr(self, section), section))

    # I/O
    # ------------------------------------------------------------------

    def write_itp(self, filename):
        """
        Write topology to file.
        Sections are written in STRUCTURED_SECTIONS order,
        followed by any unrecognized sections.
        """
        
        self._rebuild_sections()
    
        ordered_sections = list(self.STRUCTURED_SECTIONS)
        remaining_sections = [s for s in self.sections if s not in set(self.STRUCTURED_SECTIONS)]
    
        with open(filename, "w") as f:
            for section in ordered_sections + remaining_sections:
                lines = self.sections.get(section, [])
                if not lines:
                    continue
    
                f.write(f"[ {section} ]\n")
                for line in lines:
                    f.write(f"{line}\n")
                f.write("\n")

    # Merge helpers
    # ------------------------------------------------------------------

    def _normalize_line(self, line):
        return " ".join(line.split())
        
    def _check_atoms_exist(self, *names):
        """Raise ValueError if any of the given atom names are not in the topology."""
        missing = [n for n in names if n not in self.name_to_nr]
        if missing:
            raise ValueError(f"Atom(s) not found in topology: {missing}")

    def _structured_key(self, section, entry):
        """
        Return a canonical key for a section entry.
        Accounts for symmetry where applicable (bonds, angles, dihedrals, ...).
        Virtual sites are keyed by VS atom only — one definition per VS allowed.
        """
        if section == "atoms":
            return (entry["atom"],)
            
        ## clear symmetrical potentials
        if section in ["bonds", "constraints", "pairs"]:
            return tuple(sorted((entry["i"], entry["j"])))
    
        if section == "angles":
            forward = (entry["i"], entry["j"], entry["k"])
            reverse = (entry["k"], entry["j"], entry["i"])
            return min(forward, reverse)
    
        if section == "dihedrals":
            forward = (entry["i"], entry["j"], entry["k"], entry["l"])
            reverse = (entry["l"], entry["k"], entry["j"], entry["i"])
            return min(forward, reverse)
    
        if section in ("virtual_sites1", "virtual_sites2", "virtual_sites3", "virtual_sites4"):
            return (entry["i"],)  # i is the vs atom

        if section == "virtual_sitesn":
            return (entry["vs"],)
    
        if section == "exclusions":
            return tuple(sorted(entry))
    
        if isinstance(entry, str):
            return tuple(self._normalize_line(entry).split())
    
        return tuple(entry)

    def _get_structured_section(self, section):
        """Return the structured list for a named section."""
        if section in self.STRUCTURED_SECTIONS:
            return getattr(self, section, self.other_sections.get(section, []))
        return self.other_sections.get(section, [])
    
    def _set_structured_section(self, section, value):
        """Set the structured list for a named section."""
        if section in self.STRUCTURED_SECTIONS:
            setattr(self, section, value)
        else:
            self.other_sections[section] = value

    # Public API
    # ------------------------------------------------------------------

    def set_resname(self, resname):
        """
        Rename the molecule and all residue fields to resname.
        Creates a minimal moleculetype entry if none exists.
        """
        for atom in self.atoms:
            atom["residue"] = resname
            
        if self.moleculetype:
            self.moleculetype["name"] = resname
        else:
            self.moleculetype = {"name": resname, "nrexcl": "1", "extra": []}
        self._rebuild_sections()  

    def add_atom(self, name, atom_type, charge, mass=None):
        """
        Append a new atom to the topology.
        resnr and residue are inherited from the existing atoms.
        Raises ValueError if atom name already exists.
        """
        if name in self.name_to_nr:
            raise ValueError(f"Atom name '{name}' already exists.")
    
        if self.atoms:
            resnr   = self.atoms[0]["resnr"]
            residue = self.atoms[0]["residue"]
        elif self.moleculetype:
            resnr   = "1"
            residue = self.moleculetype["name"]
        else:
            resnr   = "1"
            residue = "MOL"
    
        self.atoms.append({
            "nr":      "0",
            "type":    atom_type,
            "resnr":   resnr,
            "residue": residue,
            "atom":    name,
            "cgnr":    "0",
            "charge":  str(charge),
            "mass":    str(mass) if mass is not None else None,
            "extra":   [],
        })
        self._rebuild_sections()
        return self
    
    def add_entry(self, section, *atom_names, params=None):
        """
        Add an entry to any structured section except atoms.
    
        Parameters
        ----------
        section : str
            Section name, e.g. 'bonds', 'angles', 'dihedrals', 'pairs', etc.
        *atom_names : str
            Atom names in order (i, j, k, ...).
        params : list, optional
            Interaction parameters.
    
        Examples
        --------
        >>> top.add_entry("bonds", "BB", "SC1", params=["1", "0.47", "3800"])
        >>> top.add_entry("angles", "BB", "SC1", "SC2", params=["2", "100", "25"])
        >>> top.add_entry("exclusions", "BB", "SC1")
        >>> top.add_entry("virtual_sites2", "VS", "BB", "SC1", params=["1", "0.5"])
        """
        params = params or []
        self._check_atoms_exist(*atom_names)
    
        if section in self.NBODY_SECTIONS:
            n = self.NBODY_SECTIONS[section]
            if len(atom_names) != n:
                raise ValueError(f"[ {section} ] expects {n} atoms, got {len(atom_names)}")
            labels = [chr(ord("i") + idx) for idx in range(n)]
            entry = dict(zip(labels, atom_names))
            entry["params"] = params
            getattr(self, section).append(entry)
    
        elif section == "virtual_sitesn":
            vs, site_type, *rest = atom_names
            if site_type == "3":
                # rest is expected as alternating ref_name, weight, ref_name, weight, ...
                if len(rest) % 2 != 0:
                    raise ValueError(
                        "virtual_sitesn type 3 expects alternating ref_name/weight pairs.")
                refs = [{"ref": rest[i], "weight": rest[i + 1]}
                        for i in range(0, len(rest), 2)]
                self._check_atoms_exist(*[r["ref"] for r in refs])
            else:
                refs = list(rest)
                self._check_atoms_exist(*refs)
            self.virtual_sitesn.append({"vs": vs, "type": site_type, "refs": refs})
    
        elif section == "exclusions":
            self.exclusions.append(list(atom_names))
    
        elif section == "atoms":
            raise ValueError("Use add_atom() to add atoms.")
    
        else:
            raise ValueError(f"Unknown or unsupported section: {section!r}")
    
        self._rebuild_sections()
        return self

    def remove_atom(self, name):
        """
        Remove an atom and all entries referencing it from every section.
        Raises ValueError if the atom does not exist.
    
        Examples
        --------
        >>> top.remove_atom("SC1")
        """
        if name not in self.name_to_nr:
            raise ValueError(f"Atom '{name}' not found in topology.")
            
        # remove from atoms
        self.atoms = [a for a in self.atoms if a["atom"] != name]
        
        # remove all entries referencing this atom from nbody sections
        for section in self.NBODY_SECTIONS:
            entries = self._get_structured_section(section)
            labels = [chr(ord("i") + idx) for idx in range(self.NBODY_SECTIONS[section])]
            setattr(self, section, [e for e in entries
                                     if name not in (e[l] for l in labels)])
        self.virtual_sitesn = [e for e in self.virtual_sitesn
                               if e["vs"] != name and name not in (
                                   (r["ref"] if isinstance(r, dict) else r)
                                   for r in e["refs"]
                               )]
        self.exclusions = [e for e in self.exclusions if name not in e]
        
        # rebuild
        self._rebuild_sections()
        return self
    
    def remove_entry(self, section, *atom_names):
        """
        Remove an entry from a structured section.
    
        Parameters
        ----------
        section : str
            Section name, e.g. 'bonds', 'angles', 'exclusions', etc.
        *atom_names : str
            Atom names identifying the entry to remove.
            Order does not matter for symmetric sections (bonds, angles, dihedrals).
    
        Raises ValueError if the entry is not found.
    
        Examples
        --------
        >>> top.remove_entry("bonds", "BB", "SC1")
        >>> top.remove_entry("angles", "BB", "SC1", "SC2")
        >>> top.remove_entry("exclusions", "BB", "SC1")
        """
        if section == "atoms":
            raise ValueError("Use remove_atom() to remove atoms.")
    
        if section in self.NBODY_SECTIONS:
            labels = [chr(ord("i") + idx) for idx in range(len(atom_names))]
            target = dict(zip(labels, atom_names))
        elif section == "virtual_sitesn":
            vs, *refs = atom_names
            target = {"vs": vs, "refs": list(refs)}  
        elif section == "exclusions":
            target = list(atom_names)
        else:
            raise ValueError(f"Unknown or unsupported section: {section!r}")
    
        target_key = self._structured_key(section, target)
        entries = self._get_structured_section(section)
    
        for i, entry in enumerate(entries):
            if self._structured_key(section, entry) == target_key:
                entries.pop(i)  
                self._rebuild_sections()
                return self
    
        raise ValueError(f"Entry {atom_names} not found in [ {section} ].")
    
    def merge(self, other_parser,
              remove_redundant=True,
              keep_second=False):
        """
        Merge another ITPTopology into this one.
    
        Parameters
        ----------
        other_parser : ITPTopology
            Parser to merge into self.
        remove_redundant : bool, default True
            If True, skip duplicate entries based on section-specific keys.
            Atoms are always deduplicated by name regardless of this flag.
            remove_redundant=False only affects bonded sections.
        keep_second : bool, default False
            If True and remove_redundant is True, entries from other_parser
            replace colliding entries from self.
    
        Notes
        -----
        Atoms are always deduplicated by name. keep_second controls which
        atom wins on collision. remove_redundant only affects bonded sections.
        """
        if other_parser.moleculetype:
            if not self.moleculetype or keep_second:
                self.moleculetype = deepcopy(other_parser.moleculetype)
    
        for section in self.STRUCTURED_SECTIONS[1:]:
            current = self._get_structured_section(section)
            incoming = other_parser._get_structured_section(section)
    
            if section == "atoms" and not remove_redundant:
                warnings.warn(
                    "remove_redundant=False has no effect on [ atoms ] — "
                    "atom names must be unique. Atoms will always be deduplicated. "
                    "Use keep_second to control which atom wins on collision.",
                    UserWarning, stacklevel=2)
    
            effective_remove_redundant = True if section == "atoms" else remove_redundant
    
            if keep_second and effective_remove_redundant:
                incoming_keys = {self._structured_key(section, x) for x in incoming}
                merged = [x for x in current
                          if self._structured_key(section, x) not in incoming_keys]
                merged.extend(deepcopy(x) for x in incoming)
            else:
                merged = list(current)
                current_keys = {self._structured_key(section, x) for x in merged}
                for x in incoming:
                    key = self._structured_key(section, x)
                    if not effective_remove_redundant or key not in current_keys:
                        merged.append(deepcopy(x))
                        current_keys.add(key)
    
            self._set_structured_section(section, merged)
    
        for section, lines in other_parser.other_sections.items():
            if section not in self.other_sections:
                self.other_sections[section] = deepcopy(lines)
                continue
    
            if keep_second and remove_redundant:
                incoming_keys = {self._structured_key(section, line) for line in lines}
                kept = [line for line in self.other_sections[section]
                        if self._structured_key(section, line) not in incoming_keys]
                kept.extend(deepcopy(lines))
                self.other_sections[section] = kept
            else:
                current_keys = {self._structured_key(section, line)
                                for line in self.other_sections[section]}
                for line in lines:
                    key = self._structured_key(section, line)
                    if not remove_redundant or key not in current_keys:
                        self.other_sections[section].append(deepcopy(line))
                        current_keys.add(key)
    
        self._refresh_maps()
        self._rebuild_sections()
    
        return self

    def __repr__(self):
        parts = [f"{len(self.atoms)} atoms"]
        for section in self.NBODY_SECTIONS:
            parts.append(f"{len(getattr(self, section))} {section}")
        parts.append(f"{len(self.virtual_sitesn)} virtual_sitesn")
        parts.append(f"{len(self.exclusions)} exclusions")
        return f"<ITPTopology {' | '.join(parts)}>"
        