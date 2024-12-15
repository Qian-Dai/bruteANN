import numpy as np

def read_Ivecs(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        vectors = []
        offset = 0

        while offset < len(data):
            # Read the dimension of the vector (k)
            k = np.frombuffer(data, dtype=np.int32, count=1, offset=offset)[0]
            offset += 4  # Skip dimension info

            # Read the vector identifiers (k integers)
            vector = np.frombuffer(data, dtype=np.int32, count=k, offset=offset)
            vectors.append(vector)
            offset += k * 4  # Skip the vector data

        return np.vstack(vectors)

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error while reading groundtruth file: {e}")








