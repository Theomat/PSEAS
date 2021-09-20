from typing import Dict, Optional, Tuple, List

import numpy as np

def prepare_dataset(features: Dict[str, np.ndarray], results: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], float]:
    # Fill missing features with mean feature
    # Contains what's to fill
    to_fill: List[Tuple[str, Optional[np.np.ndarray]]] = []
    # Contains the sum of each feature that is not missing
    total_feature: np.ndarray = None
    # Contains the number of each feature that is not missing
    counts: np.ndarray = None

    for instance in results.keys():
        if not instance in features:
            to_fill.append((instance, None))
        else:
            feature = features[instance]
            missing: np.ndarray = np.isnan(feature)
            mask: np.ndarray = np.logical_not(missing)
            # Late initialisation to get the right array size
            if total_feature is None:
                total_feature = np.zeros_like(feature)
                counts = np.zeros_like(total_feature)
            
            total_feature[mask] += feature[mask]
            counts += mask
            if np.any(missing):
                to_fill.append((instance, missing))
    # Now total_feature will contain average feature
    total_feature /= counts
    # Fill missings
    for instance, mask in to_fill:
        if mask is None:
            features[instance] = total_feature.copy()
        else:
            (features[instance])[mask] = total_feature[mask]
    # Compute max time
    times: np.ndarray = np.array([list(v.values()) for v in results.values()])
    max_time: float = np.max(times)
    return features, results, max_time


# ==========================================================================================================
# TESTING
# ==========================================================================================================
if __name__ == "__main__":
    pass
