# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch_geometric
from torch_geometric.data import Data


def get_pyg_data_keys(event: Data):
    """
    Get the keys of the pyG data object.
    """
    if torch_geometric.__version__ < "2.4.0":
        return event.keys
    else:
        return event.keys()
