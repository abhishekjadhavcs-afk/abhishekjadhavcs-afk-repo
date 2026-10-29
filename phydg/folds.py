"""Region clusters for leave-one-cluster-out (LOCO) evaluation on Sen12Landslides S12LS-LD (S2).

Inventories are grouped by geography so that a held-out cluster shares no
neighbouring terrain with the training clusters. The Sen12Landslides paper uses
its own 6-cluster grouping; replace CLUSTERS with that grouping once confirmed
from the paper so results are directly comparable.
"""

CLUSTERS = {
    "europe": ["italy"],
    "central_asia": ["kyrgyzstan1", "kyrgyzstan2"],
    "east_asia": ["hiroshima", "hokkaido", "china"],
    "southeast_asia": ["indonesia", "itogon"],
    "south_asia": ["thrissur"],
    "africa": ["chimanimani"],
    "caribbean": ["dominicamaria", "usa"],
    "oceania": ["newzealand"],
}

INVENTORY_TO_CLUSTER = {inv: c for c, invs in CLUSTERS.items() for inv in invs}
INVENTORIES = sorted(INVENTORY_TO_CLUSTER)
INVENTORY_INDEX = {inv: i for i, inv in enumerate(INVENTORIES)}
