# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
import os, shutil
import numpy as np
import pandas as pd

region_labels={
    1: {"hardware": "PIXEL", "barrel_endcap": -2},
    2: {"hardware": "STRIP", "barrel_endcap": -2},
    3: {"hardware": "PIXEL", "barrel_endcap": 0},
    4: {"hardware": "STRIP", "barrel_endcap": 0},
    5: {"hardware": "PIXEL", "barrel_endcap": 2},
    6: {"hardware": "STRIP", "barrel_endcap": 2},
}

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))


def add_region_labels(hits, region_labels: dict):
    """
        Label the 6 detector regions (forward-endcap pixel, forward-endcap strip, etc.)
        """
        
    for region_label, conditions in region_labels.items():
        condition_mask = np.logical_and.reduce([hits[condition_column] == condition for condition_column, condition in conditions.items()])
        hits.loc[condition_mask, "region"] = region_label

    assert (hits.region.isna()).sum() == 0, "There are hits that do not belong to any region!"

    return hits

event_data_dir = "/sps/l2it/scaillou/data/TEST_COMMON_FRAMEWORK/APPROVAL_SUMMER_2022/feature_store/testset"
output_dir = "/sps/l2it/scaillou/data/TEST_COMMON_FRAMEWORK/APPROVAL_SUMMER_2022/data/testset"

for event_file in os.listdir(event_data_dir):
    if "-truth" in event_file:
        hits = pd.read_csv(os.path.join(event_data_dir, event_file))
        hits = add_region_labels(hits, region_labels)
        hits.to_csv(os.path.join(output_dir, event_file), index=False)
    if "-particle" in event_file:
        shutil.copy(os.path.join(event_data_dir, event_file), output_dir)