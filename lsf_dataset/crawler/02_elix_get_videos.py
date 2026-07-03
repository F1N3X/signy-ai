import requests
import re

def get_word_videos(word):
    page_url = f"https://dico.elix-lsf.fr/dictionnaire/{word}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()
        
        # On extrait le(s) word_id grâce à un regex
        # Le pattern cherche '"word_id":' suivi de chiffres
        word_ids = re.findall(r'"word_id"\s*:\s*(\d+)', response.text)
        
        if not word_ids:
            print(f"❌ Aucun ID trouvé pour le mot '{word}'.")
            return
            
        # On garde les IDs uniques (au cas où il y en aurait plusieurs identiques)
        unique_ids = list(set(word_ids))
        print(f"🔗 ID(s) trouvé(s) pour '{word}' : {unique_ids}")
        
        target_id = unique_ids[0]
        api_url = f"https://api.elix-lsf.fr/words/{word}/meanings/{target_id}"
        
        api_response = requests.get(api_url, headers=headers)
        api_response.raise_for_status()
        data = api_response.json()
        
        # 4. Extraction des vidéos
        word_signs = data.get("wordSigns", [])
        if word_signs:
            print(f"\n🎥 Vidéos pour '{word}' :")
            for sign in word_signs:
                print(f"   - {sign.get('uri')}")
        else:
            print(f"ℹ️ Pas de vidéo wordSigns trouvée pour l'ID {target_id}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors de la requête : {e}")

# Test du script
if __name__ == "__main__":
    get_word_videos("a")