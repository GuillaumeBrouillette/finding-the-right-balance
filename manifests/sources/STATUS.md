# Source-document group manifest status

`manifests/source_groups/` freezes source-document group
IDs for the seven core datasets plus the reported early BEIR ArguAna and
Webis-Touche2020 tasks. The recovered source files are checksummed, and their
query order is required to match the exact query IDs in `manifests/splits/`.

The historical result CSVs still do not contain the final per-query dense
candidate pools, selected rankings, injected-copy parents, or chunk parents.
Those are execution artifacts and are not fabricated by the source manifests.

Future runs must persist, for every candidate instance:

```text
dataset, query_id, candidate_id, source_group_id, parent_candidate_id,
transformation, rank
```

Injected copies and overlapping chunks must retain the source group's ID and
record their original candidate as the parent.
