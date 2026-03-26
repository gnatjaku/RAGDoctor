Instrukcja: instalacja Dockera, pobranie i uruchomienie MongoDB w Dockerze

1) Sprawdź dystrybucję (Ubuntu/Debian) — w twoim systemie wykryto Ubuntu 24.04.

2) Instalacja Dockera (szybkie polecenia dla Ubuntu/Debian):

```bash
# usuń stare wersje
sudo apt remove docker docker-engine docker.io containerd runc -y || true

# wymagane pakiety
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release -y

# dodaj repo Docker
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y

# uruchom i włącz przy starcie
sudo systemctl enable --now docker

# (opcjonalnie) daj swojemu użytkownikowi prawo do uruchamiania docker bez sudo
sudo usermod -aG docker $USER
# potem wyloguj się i zaloguj ponownie
```

3) Uruchomienie MongoDB z docker-compose

- Skopiuj `.env.example` do `.env` i ustaw silne hasło.

```bash
cp .env.example .env
# Edytuj .env i ustaw MONGO_INITDB_ROOT_PASSWORD

# Uruchom usługę
docker compose up -d
```

4) Weryfikacja

```bash
docker ps
docker logs mongodb --tail 200
# połącz się z shell mongo
docker exec -it mongodb mongosh -u "$MONGO_INITDB_ROOT_USERNAME" -p "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin
```

5) Backup i przywracanie (szybkie):

```bash
# backup folderu wolumenu
docker run --rm -v mongo-data:/data/db -v $(pwd):/backup ubuntu tar czf /backup/mongo-data-backup.tar.gz -C /data/db .

# przywracanie
docker run --rm -v mongo-data:/data/db -v $(pwd):/backup ubuntu bash -c "cd /data/db && tar xzf /backup/mongo-data-backup.tar.gz --strip 1"
```

Uwaga: Nie wystawiaj portu 27017 do internetu bez zabezpieczeń. Rozważ firewall, VPN, lub uruchomienie tylko w sieci lokalnej.
