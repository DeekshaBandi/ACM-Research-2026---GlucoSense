import hashlib
import os

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(1048576), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        return None

def verify_files(data_folder, checksum_file):
    print(f"--- Verification Started ---")
    print(f"Checking folder: {data_folder}")
    
    if not os.path.exists(checksum_file):
        print(f"❌ ERROR: Checksum file not found at: {checksum_file}")
        return

    with open(checksum_file, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    print(f"Found {len(lines)} files to check. Starting...\n")

    for line in lines:
        if line.startswith("[source"): continue # Skip markers
            
        parts = line.split()
        # Handle both standard and 'source-prefixed' formats
        expected_hash = parts[2] if parts[0].startswith("[") else parts[0]
        relative_path = parts[3] if parts[0].startswith("[") else parts[1]
        
        local_file_path = os.path.join(data_folder, relative_path.replace('/', os.sep))

        if os.path.exists(local_file_path):
            actual_hash = calculate_sha256(local_file_path)
            if actual_hash == expected_hash:
                print(f"✅ OK: {relative_path}")
            else:
                print(f"❌ CORRUPT: {relative_path}")
        else:
            print(f"⚠️  MISSING: {relative_path}")

if __name__ == "__main__":
    # DOUBLE CHECK THIS PATH: Is your 34GB data here? 
    # Or is it still in C:\Users\ericl\DESTINATION?
    base_dir = r"C:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data\glycemic_data"
    checksum_path = os.path.join(base_dir, "SHA256SUMS.txt")
    
    verify_files(base_dir, checksum_path)