import string
import json
import time
import requests
import os

BASE_URL = "https://api.elix-lsf.fr/suggests"

def fetch_all_words():
    alphabet_dict = {}
    
    for letter in string.ascii_lowercase:
        print(f"Fetching letter : {letter.upper()}")
        alphabet_dict[letter] = []
        
        offset = 0
        limit = 1000
        
        while True:
            params = {
                'q': letter,
                'limit': limit,
                'offset': offset
            }
            
            try:
                response = requests.get(BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                words = data.get("data", [])
                total_mots = data.get("total", 0)
                
                if not words:
                    break

                alphabet_dict[letter].extend(words)
                print(f"   -> {len(alphabet_dict[letter])} / {total_mots} words fetched...")
                
                if len(alphabet_dict[letter]) >= total_mots or len(words) < limit:
                    break
                
                offset += limit
                
                time.sleep(0.3)
                
            except requests.exceptions.RequestException as e:
                print(f"Error while fetching words for letter {letter} (offset {offset}): {e}")
                break
                
    return alphabet_dict

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    full_dictionary = fetch_all_words()
    
    output_dir = "lsf_dataset/metadata"
    os.makedirs(output_dir, exist_ok=True)

    filename = "elix_full_dictionary.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(full_dictionary, f, ensure_ascii=False, indent=4)
        
    end_time = time.perf_counter()
    print(f"\nCompleted in {end_time - start_time:.2f} seconds !")
    print(f"All data has been saved to : '{filepath}'")