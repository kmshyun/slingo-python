import h5py

def read_all_h5(file_path):
    def recurse(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"[Group] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"[Dataset] {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            try:
                data = obj[:]
                print(data)
            except Exception as e:
                print(f"  [Error reading data: {e}]")

    with h5py.File(file_path, 'r') as f:
        print(f"Reading file: {file_path}")
        f.visititems(recurse)

# 사용 예시
read_all_h5('gesture_classifier.h5')