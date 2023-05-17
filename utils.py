def map_keys_to_json(d):
    return [{'key': k, 'value': v} for (k, v) in d.items()]

def inverse_map_json(l):
    return {(tuple(d['key'][0]), d['key'][1]):d['value'] for d in l}

def list_from_sums(D):
    return sorted([v for v, sm in D.items() for _ in range(sm)])
