from pathlib import Path
import string

TAIL_CODES = {
    "T": "cC",      # C08:0 octanoyl
    "J": "CC",      # C10:0 decanoyl
    "U": "cCC",     # C12:0 lauroyl
    "M": "CCC",     # C14:0 myristoyl
    "P": "cCCC",    # C16:0 palmitoyl
    "S": "CCCC",    # C18:0 stearoyl
    "K": "cCCCC",   # C20:0 arachidoyl
    "B": "CCCCC",   # C22:0 behenoyl
    "X": "cCCCCC",  # C24:0 lignoceroyl
    "H": "CCCCCC",  # C26:0 hexacosanoyl
    "R": "CDC",     # C14:1(9c) myristoleoyl
    "Y": "cCDC",    # C16:1(9c) palmitoleoyl
    "O": "CDCC",    # C18:1(9c) oleoyl
    "G": "cCDCC",   # C20:1(11c) gondoic
    "E": "CCDCC",   # C22:1 erucoyl
    "N": "cCCDCC",  # C24:1 nervonic
    "V": "CCDC",    # C18:1(11c) cis-vaccenic
    "L": "CDDC",    # C18:2 linoleoyl
    "F": "CDDD",    # C18:3 alpha-linolenic
    "I": "cCDDC",   # C20:2 eicosadienoyl
    "Q": "cDDDC",   # C20:3 eicosatrienoyl
    "A": "cFFDC",   # C20:4 arachidonoyl
    "DHA": "DFFDD",   # C22:6 DHA
}

linkerMapp = {
    # beadtype, beadname, charge, ffbonded name, bonded suffix
    "G":   ["SN4a",  "GL?",  "0",  "GL",  "glyc"],
    "L":   ["SN4ah", "PL?",  "0",  "ET",  "plasm"],
    "E":   ["SN3a",  "ET?",  "0",  "ET",  "ether"],
    "A1":  ["SP1",   "OH1",  "0",  "OH1", "sm"],
    "A2":  ["SP2",   "AM2",  "0",  "AM2", "sm"],
    "FA":  ["SQ5n",  "COO", "-1",  "COO", "fa"],
    ## Special Linkers for Mariana's work
    "O1":   ["SP2",  "O1",  "0",  "GL",  "glyc"],
    "O2":   ["SN4a",  "O2",  "0",  "GL",  "glyc"],

}

tailMapp = {
    # beadtype, beadname, charge, bondname, angle/dih name
    "C":  ["C1",   "C??",   "0",    "C1",    "C1"],
    "c":  ["SC1",  "C??",   "0",   "SC1",    "C1"],
    "D":  ["C4h",  "D??",   "0",    "C4",    "C4"],
    "d":  ["SC4h", "D??",   "0",   "SC4",    "C4"],
    "F":  ["C5h",  "D??",   "0",    "C4",    "C4"],
    "T":  ["C4h",  "T??",   "0",    "C4",    "C1"],
    "t":  ["SC4h", "T??",   "0",   "SC4",    "C1"],
    "W":  ["C6r",  "W??",   "0",    "C1",    "C1"],
    "w":  ["SC6r", "W??",   "0",   "SC1",    "C1"],
}

CHAIN_INDEX = {c: i + 1 for i, c in enumerate(string.ascii_uppercase)}


def _expand_tail_code(tail_spec):
    """Expand one-letter tail aliases such as T -> cC."""
    return TAIL_CODES.get(tail_spec, tail_spec)


def _parse_spec(spec):
    """
    Parse strings like:
        G-cC
        G-T
        E-CDCC
        A1-tCCC
        cC
        T
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty tail specification.")

    if "-" in spec:
        linker_key, tail_spec = spec.split("-", 1)
        linker_key = linker_key.strip()
        tail_spec = tail_spec.strip()

        if linker_key not in linkerMapp:
            raise ValueError(f"Unsupported linker '{linker_key}'. Allowed: {list(linkerMapp)}")
        if not tail_spec:
            raise ValueError("Missing tail after linker.")

        return linker_key, _expand_tail_code(tail_spec)

    return None, _expand_tail_code(spec)


def _validate_tail_string(tail_string):
    invalid = [x for x in tail_string if x not in tailMapp]
    if invalid:
        raise ValueError(f"Unsupported tail code(s): {invalid}. Allowed: {list(tailMapp)}")


def _tail_bond_name(prev_code, curr_code, bead_pos, tail_len):
    """
    Bond naming for internal tail bonds, following the same logic as the Martini lipid builder:
    - second bond (i == 2) gets *_mid_5long if both beads are regular-sized
    - last bond gets *_end
    - otherwise *_mid
    """
    prev_bond_code = tailMapp[prev_code][3]
    curr_bond_code = tailMapp[curr_code][3]

    prev_small = tailMapp[prev_code][0].startswith("S")
    curr_small = tailMapp[curr_code][0].startswith("S")

    if bead_pos == 2 and (not prev_small) and (not curr_small):
        return f"b_{prev_bond_code}_{curr_bond_code}_mid_5long"
    elif bead_pos == tail_len:
        return f"b_{prev_bond_code}_{curr_bond_code}_end"
    else:
        return f"b_{prev_bond_code}_{curr_bond_code}_mid"


def build_tail_itp(spec, 
                   molname=None, resname=None, out=None,
                   nrexcl = 1, chain_id = "A"):
    """
    Build a Martini 3 topology (.itp) for a single acyl chain with an optional linker bead.

    This function generates the [moleculetype], [atoms], [bonds], and [angles]
    sections for a single lipid tail, optionally attached to a linker bead
    (e.g., glycerol, ether, sphingosine, or free fatty acid).

    The tail can be provided either explicitly (e.g., "cC", "CDCC") or using
    shorthand codes (e.g., "T" for octanoyl, "O" for oleoyl), which are internally
    expanded.

    Parameters
    ----------
    spec : str
        Tail specification string. Supported formats include:
        - "G-cC"      : linker + explicit tail
        - "G-T"       : linker + shorthand tail (T → cC)
        - "E-CDCC"    : linker + unsaturated tail
        - "A1-tCCC"   : sphingosine-type linker + tail
        - "cC"        : tail only (no linker)
        - "T"         : shorthand tail only

        Linker and tail are separated by a dash ("-"). If no linker is given,
        only the tail is built.

    molname : str, optional
        Molecule name used in the [ moleculetype ] section.
        Defaults to the spec string with '-' replaced by '_'.

    resname : str, optional
        Residue name used in the [ atoms ] section.
        Defaults to the first 4 characters of molname.

    out : str or Path, optional
        Output file path. If provided, the generated .itp file is written to disk.
        If None, the topology is only returned as a string.

    nrexcl : int, default=1
        Number of exclusions used in the [ moleculetype ] section.

    chain_id : str, default="A"
        Chain identifier (single letter A–Z).
        This controls:
        - Tail bead names: C1A, C2A, ... / C1B, C2B, ...
        - Linker numbering: GL1, GL2, GL3, ... based on alphabetical order

        For example:
        - chain_id="A" → GL1, C1A, C2A
        - chain_id="B" → GL2, C1B, C2B

    Returns
    -------
    str
        The generated Martini 3 .itp file contents.

    Notes
    -----
    - Only a single linker bead is supported.

    - Available tail shorthand codes:
    
        code  pattern   description
        --------------------------------------------------
        T     cC        C08:0 octanoyl
        J     CC        C10:0 decanoyl
        U     cCC       C12:0 lauroyl
        M     CCC       C14:0 myristoyl
        P     cCCC      C16:0 palmitoyl
        S     CCCC      C18:0 stearoyl
        K     cCCCC     C20:0 arachidoyl
        B     CCCCC     C22:0 behenoyl
        X     cCCCCC    C24:0 lignoceroyl
        H     CCCCCC    C26:0 hexacosanoyl
        R     CDC       C14:1(9c) myristoleoyl
        Y     cCDC      C16:1(9c) palmitoleoyl
        O     CDCC      C18:1(9c) oleoyl
        G     cCDCC     C20:1(11c) gondoic
        E     CCDCC     C22:1 erucoyl
        N     cCCDCC    C24:1 nervonic
        V     CCDC      C18:1(11c) cis-vaccenic
        L     CDDC      C18:2(9c,12c) linoleoyl
        F     CDDD      C18:3(9c,12c,15c) alpha-linolenic
        I     cCDDC     C20:2(11c,14c) eicosadienoyl
        Q     cDDDC     C20:3(8c,11c,14c) eicosatrienoyl
        A     cFFDC     C20:4(5c,8c,11c,14c) arachidonoyl
        DHA   DFFDD     C22:6(4c,7c,10c,13c,16c,19c) DHA

    - Available linker definitions:
        
        key   beadtypebeadnamecharge  ffname  suffix  Notes
        ---------------------------------------------------------
        G     SN4a    GL?     0       GL      glyc    Glycerol ester linker
        L     SN4ah   PL?     0       ET      plasm   Plasmalogen linker
        E     SN3a    ET?     0       ET      ether   Ether linker
        A1    SP1     OH1     0       OH1     sm      sphingosine backbone
        A2    SP2     AM2     0       AM2     sm      sphingosine backbone
        FA    SQ5n    COO     -1      COO     fa      free fatty acid
        O1    SP2     O1      0       GL      glyc    Custom
        O2    SN4a    O2      0       GL      glyc    Custom
        
        
    - Available tail bead definitions:
        
        key   beadtype  beadname   charge  bond    angle  Notes
        ----------------------------------------------------------------------
        C     C1        C??        0       C1      C1     straight chain
        c     SC1       C??        0       SC1     C1     short straight chain
        D     C4h       D??        0       C4      C4     chain with cis double bond
        D     SC4h      D??        0       SC4     C4     short chain with cis double bond
        F     C5h       D??        0       C4      C4     chain with more than one double bond (normally 1.5)
        T     C4h       T??        0       C4      C1     chain with trans double bond (only used in shingosine top bead)
        t     SC4h      T??        0       SC4     C1     short chain with trans double bond (only used in shingosine top bead)
        W     C6r       W??        0       C1      C1     Custom bead for terminal triple bond. Only bead type changes. Not validated.
        w     SC6r      W??        0       C1      C1     Custom bead for terminal triple bond. Only bead type changes. Not validated.


    This function builds upon concepts and naming conventions from the
    Martini 3 lipid topology generator:
    
    Helgi I. Ingolfsson, Kasper B. Pedersen, and Tsjerk A. Wassenaar
    
    Please refer to the original Martini 3 lipid builder for full functionality.

    Examples
    --------
    Build a glycerol-linked octanoyl tail:

    >>> build_tail_itp("G-T")

    Build an ether-linked oleoyl tail with chain B naming:

    >>> build_tail_itp("E-O", chain_id="B")

    Build a free fatty acid:

    >>> build_tail_itp("FA-cC")

    Write output to file:

    >>> build_tail_itp("G-cC", out="tail.itp")
    """

    
    linker_key, tail_string = _parse_spec(spec)
    _validate_tail_string(tail_string)

    chain_id = chain_id.upper()
    if chain_id not in CHAIN_INDEX:
        raise ValueError("chain_id must be a single letter between A and Z")
    chain_num = CHAIN_INDEX[chain_id]

    if molname is None:
        molname = spec.replace("-", "_")
    if resname is None:
        resname = "UNK"

    atoms = []
    bonds = []
    angles = []

    idx = 1
    linker_idx = None
    tail_indices = []

    ffname = None
    suffix = None

    # Optional linker
    if linker_key is not None:
        beadtype, beadname, charge, ffname, suffix = linkerMapp[linker_key]
        if "?" in beadname:
            atomname = beadname.replace("?", str(chain_num))
        else:
            atomname = beadname
        atoms.append([idx, beadtype, 0, resname, atomname, idx, charge])
        linker_idx = idx
        idx += 1

    # Tail beads
    for i, code in enumerate(tail_string, start=1):
        beadtype, beadname, charge, _, _ = tailMapp[code]
        atomname = beadname.replace("??", f"{i}{chain_id}")
        atoms.append([idx, beadtype, 0, resname, atomname, idx, charge])
        tail_indices.append(idx)

        # bond to linker or previous tail bead
        if i == 1:
            if linker_idx is not None:
                if tailMapp[code][3].startswith("S"):
                    bname = f"b_{ffname}_{tailMapp[code][3]}_{suffix}"
                else:
                    bname = f"b_{ffname}_{tailMapp[code][3]}_{suffix}_5long"
                bonds.append([linker_idx, idx, bname])
        else:
            prev_code = tail_string[i - 2]
            bname = _tail_bond_name(prev_code, code, i, len(tail_string))
            bonds.append([idx - 1, idx, bname])
        idx += 1

    # linker-tail-tail angle
    if linker_idx is not None and len(tail_string) >= 2:
        code1 = tail_string[0]
        code2 = tail_string[1]
        aname = f"a_{ffname}_{tailMapp[code1][4]}_{tailMapp[code2][4]}_{suffix}"
        angles.append([linker_idx, tail_indices[0], tail_indices[1], aname])

    # internal tail angles
    if len(tail_string) >= 3:
        for i in range(1, len(tail_string) - 1):
            c0 = tail_string[i - 1]
            c1 = tail_string[i]
            c2 = tail_string[i + 1]
            aname = f"a_{tailMapp[c0][4]}_{tailMapp[c1][4]}_{tailMapp[c2][4]}_def"
            angles.append([tail_indices[i - 1], tail_indices[i], tail_indices[i + 1], aname])

    # Write output
    lines = []
    lines.append(f"; Single-tail Martini 3 topology generated with Mixer.\n; Requested spec: {spec}")
    lines.append(f"; Expanded tail string: {tail_string}")
    lines.append(f"; Chain ID: {chain_id}")
    lines.append("")

    lines.append("[ moleculetype ]")
    lines.append("; molname       nrexcl")
    lines.append(f"{molname:<14} {nrexcl}")
    lines.append("")

    lines.append("[ atoms ]")
    lines.append("; id  type   resnr  residu  atom  cgnr  charge")
    for a in atoms:
        lines.append(f"{a[0]:>4}  {a[1]:<6} {a[2]:>4}   {a[3]:<6} {a[4]:<4}  {a[5]:>4}  {a[6]:>6}")

    lines.append("")
    lines.append("[ bonds ]")
    lines.append(";  i   j   name")
    for b in bonds:
        lines.append(f"{b[0]:>4} {b[1]:>4}   {b[2]}")

    if angles:
        lines.append("")
        lines.append("[ angles ]")
        lines.append(";  i   j   k   name")
        for a in angles:
            lines.append(f"{a[0]:>4} {a[1]:>4} {a[2]:>4}   {a[3]}")

    text = "\n".join(lines) + "\n"

    if out is not None:
        Path(out).write_text(text)

    return text