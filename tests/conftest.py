import pytest
from mixer.parser import ITPTopology

SIMPLE_ITP = """
[ moleculetype ]
MOL 1

[ atoms ]
1 P4   1 MOL BB  1 0.0
2 SC4  1 MOL SC1 2 0.0
3 SC4  1 MOL SC2 3 0.0

[ bonds ]
1 2 1 0.47 3800
2 3 1 0.47 3800

[ angles ]
1 2 3 2 100 25

[ constraints ]
1 2 1 0.47

[ pairs ]
1 3 1

[ exclusions ]
1 2 3
"""

VS_ITP = """
[ moleculetype ]
VSM 1

[ atoms ]
1 P4   1 VSM BB  1  0.0 72
2 SC4  1 VSM SC1 2  0.0 45
3 SC4  1 VSM SC2 3  0.0 45
4 SC4  1 VSM SC3 4  0.0 45
5 VS   1 VSM VS1 5  0.0

[ bonds ]
1 2 1 0.47 3800
2 3 1 0.47 3800

[ angles ]
1 2 3 2 100 25

[ dihedrals ]
1 2 3 4 1 180 5 1

[ virtual_sites2 ]
5 1 2 1 0.5

[ virtual_sitesn ]
5 1 1 2 3

[ exclusions ]
1 2 3
"""

MINIMAL_ITP = """
[ moleculetype ]
MIN 1

[ atoms ]
1 P4 1 MIN BB 1 0.0
"""

MERGE_ITP_A = """
[ moleculetype ]
MOLA 1

[ atoms ]
1 P4  1 MOLA BB  1  0.0 72
2 SC4 1 MOLA SC1 2  0.0 45
3 SC4 1 MOLA SC2 3  0.0 45
4 SC4 1 MOLA SC3 4  0.0 45
5 VS  1 MOLA VS1 5  0.0

[ bonds ]
1 2 1 0.47 3800
2 3 1 0.47 3800

[ angles ]
1 2 4 2 100 25
1 2 3 2 100 25

[ virtual_sites2 ]
5 1 2 1 0.5

[ virtual_sitesn ]
5 1 1 2 3

[ exclusions ]
1 2
1 2 3

[ custom_section ]
shared line
unique line A
"""

MERGE_ITP_B = """
[ moleculetype ]
MOLB 1

[ atoms ]
1 P3  1 MOLB BB  1  0.0 72
2 SC4 1 MOLB SC1 2  0.0 45
3 SC4 1 MOLB SC3 3  0.0 45
4 SC4 1 MOLB SC4 4  0.0 45
5 VS  1 MOLB VS1 5  0.0
6 VS  1 MOLB VS2 6  0.0

[ bonds ]
1 2 1 0.35 2000
2 3 1 0.35 4000

[ angles ]
1 2 3 2 100 25
1 2 4 2 100 25

[ virtual_sites2 ]
5 1 2 1 0.5
6 1 3 1 0.7

[ virtual_sitesn ]
5 1 1 2 3

[ exclusions ]
1 2
1 2 3

[ custom_section ]
shared line
unique line B
"""

@pytest.fixture
def simple_top():
    return ITPTopology(SIMPLE_ITP)

@pytest.fixture
def vs_top():
    return ITPTopology(VS_ITP)

@pytest.fixture
def minimal_top():
    return ITPTopology(MINIMAL_ITP)

@pytest.fixture
def merge_top_a():
    return ITPTopology(MERGE_ITP_A)

@pytest.fixture
def merge_top_b():
    return ITPTopology(MERGE_ITP_B)