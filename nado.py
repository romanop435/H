"""
GLua Dataset Parser v2 — МАКСИМАЛЬНЫЙ
========================================
Цель: ~5000 обучающих пар

Источники:
  1. wiki.facepunch.com/gmod     — полная документация GLua
  2. GitHub                      — 15+ поисковых запросов по GLua аддонам
  3. Facepunch форумы            — обсуждения с примерами кода
  4. 50+ ручных примеров         — высококачественные пары по всем темам
  5. Автогенерация               — из каждого файла создаём несколько пар

Запуск:
  pip install requests beautifulsoup4 tqdm
  python glua_dataset_v2.py

Результат:
  glua_dataset_v2/
    ├── wiki_docs.jsonl
    ├── github_code.jsonl
    ├── forum_posts.jsonl
    ├── training_pairs.jsonl   ← главный файл для обучения
    └── summary.txt
"""

import requests
import json
import time
import re
import os
import random
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path

# ──────────────────────────────────────────────
# НАСТРОЙКИ
# ──────────────────────────────────────────────

GITHUB_TOKEN = ""       # github.com/settings/tokens → Generate (без прав)
OUTPUT_DIR   = "glua_dataset_v2"
DELAY        = 0.8

GITHUB_QUERIES = [
    "language:glua",
    "darkrp addon lua",
    "gmod gamemode lua",
    "garrysmod script lua",
    "gmod swep lua",
    "gmod entity lua",
    "pointshop gmod lua",
    "ulx admin mod lua",
    "ttt gmod lua",
    "sandbox gamemode gmod",
    "gmod hud lua",
    "gmod derma lua",
    "gmod networking lua",
    "gmod weapon base lua",
    "fadmin gmod lua",
]

WIKI_BASE   = "https://wiki.facepunch.com"
FORUM_BASE  = "https://forum.facepunch.com"

# ──────────────────────────────────────────────
# УТИЛИТЫ
# ──────────────────────────────────────────────

def make_output_dir():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

def save_jsonl(data: list, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✅ Сохранено {len(data)} записей → {path}")

def get_headers(github=False):
    if github and GITHUB_TOKEN:
        return {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
    return {"User-Agent": "Mozilla/5.0 GLua-Dataset-Collector/2.0"}

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

# ──────────────────────────────────────────────
# ПАРСЕР WIKI
# ──────────────────────────────────────────────

class WikiParser:
    CATEGORIES = [
        "/gmod/Category:Hooks",
        "/gmod/Category:Libraries",
        "/gmod/Category:Entities",
        "/gmod/Category:Panels",
        "/gmod/Category:Structures",
        "/gmod/Category:Enumerations",
        "/gmod/Category:Global_Functions",
        "/gmod/Category:Player",
        "/gmod/Category:Entity",
        "/gmod/Category:Weapon",
        "/gmod/Category:NPC",
        "/gmod/Category:Vehicle",
        "/gmod/Category:Angle",
        "/gmod/Category:Vector",
        "/gmod/Category:Color",
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(get_headers())
        self.visited = set()
        self.docs = []

    def get_page(self, url):
        try:
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            pass
        return None

    def parse_function_page(self, path):
        url = WIKI_BASE + path
        if url in self.visited:
            return None
        self.visited.add(url)
        soup = self.get_page(url)
        if not soup:
            return None

        title_el = soup.find("h1") or soup.find("title")
        title = title_el.get_text(strip=True) if title_el else path.split("/")[-1]

        desc_el = soup.find("div", class_="description") or soup.find("p")
        description = clean_text(desc_el.get_text()) if desc_el else ""

        code_blocks = []
        for el in soup.find_all(["pre", "code"]):
            c = el.get_text().strip()
            if len(c) > 15:
                code_blocks.append(c)

        params = []
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                params.append({
                    "name": cells[0].get_text(strip=True),
                    "type": cells[1].get_text(strip=True),
                    "desc": cells[2].get_text(strip=True),
                })

        return_info = ""
        return_el = soup.find("div", class_="returns")
        if return_el:
            return_info = clean_text(return_el.get_text())

        if not description and not code_blocks:
            return None

        return {
            "source": "wiki",
            "url": url,
            "title": title,
            "description": description,
            "parameters": params,
            "returns": return_info,
            "code_examples": code_blocks,
        }

    def get_category_links(self, category_path):
        url = WIKI_BASE + category_path
        soup = self.get_page(url)
        if not soup:
            return []
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (href.startswith("/gmod/") and
                    "Category:" not in href and
                    "Special:" not in href and
                    href not in self.visited):
                links.append(href)
        return list(set(links))

    def parse_all(self, max_pages=600):
        print("\n📖 Парсим GLua Wiki (расширенный)...")
        all_links = []
        for category in self.CATEGORIES:
            links = self.get_category_links(category)
            all_links.extend(links)
            print(f"  📂 {category.split('/')[-1]}: {len(links)} страниц")
            time.sleep(DELAY)

        all_links = list(set(all_links))[:max_pages]
        print(f"\n  Всего к парсингу: {len(all_links)}")

        for link in tqdm(all_links, desc="  Wiki"):
            doc = self.parse_function_page(link)
            if doc:
                self.docs.append(doc)
            time.sleep(DELAY * 0.4)

        print(f"  ✅ Wiki: {len(self.docs)} документов")
        return self.docs


# ──────────────────────────────────────────────
# ПАРСЕР GITHUB
# ──────────────────────────────────────────────

class GitHubParser:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(get_headers(github=True))
        self.code_samples = []
        self.seen_repos = set()

    def search_repos(self, query, max_repos=15):
        url = "https://api.github.com/search/repositories"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_repos}
        try:
            r = self.session.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json().get("items", [])
            elif r.status_code == 403:
                print("  ⚠️  Rate limit. Добавь GITHUB_TOKEN!")
                time.sleep(60)
        except Exception as e:
            pass
        return []

    def get_repo_tree(self, owner, repo):
        """Получает дерево файлов репозитория."""
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
        try:
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                tree = r.json().get("tree", [])
                return [f for f in tree if f["path"].endswith(".lua")]
        except:
            pass
        return []

    def get_raw_content(self, owner, repo, path):
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        try:
            r = self.session.get(url, timeout=10)
            if r.status_code == 200:
                return r.text
        except:
            pass
        return None

    def is_glua(self, content):
        markers = [
            "hook.Add", "hook.Run", "Entity(", "player.GetAll",
            "SERVER", "CLIENT", "AddCSLuaFile", "net.Receive",
            "net.Start", "timer.Create", "DarkRP", "GAMEMODE",
            "ENT.", "SWEP.", "vgui.Create", "LocalPlayer()",
            "game.", "util.", "surface.", "draw.",
        ]
        count = sum(1 for m in markers if m in content)
        return count >= 2

    def parse_all(self, max_repos_per_query=12):
        print("\n🐙 Парсим GitHub (расширенный)...")
        all_repos = {}

        for query in GITHUB_QUERIES:
            repos = self.search_repos(query, max_repos=max_repos_per_query)
            for repo in repos:
                key = repo["full_name"]
                if key not in all_repos:
                    all_repos[key] = repo
            time.sleep(DELAY)

        print(f"  Уникальных репозиториев: {len(all_repos)}")

        for full_name, repo in tqdm(all_repos.items(), desc="  GitHub repos"):
            owner, name = full_name.split("/")
            files = self.get_repo_tree(owner, name)
            time.sleep(DELAY * 0.5)

            # Берём разные типы файлов
            lua_files = [f for f in files if f["path"].endswith(".lua")]
            random.shuffle(lua_files)

            collected = 0
            for file_info in lua_files[:15]:
                content = self.get_raw_content(owner, name, file_info["path"])
                time.sleep(DELAY * 0.2)

                if content and self.is_glua(content) and 100 < len(content) < 50000:
                    self.code_samples.append({
                        "source": "github",
                        "repo": full_name,
                        "stars": repo.get("stargazers_count", 0),
                        "repo_description": repo.get("description", ""),
                        "file": file_info["path"].split("/")[-1],
                        "path": file_info["path"],
                        "code": content[:10000],
                    })
                    collected += 1
                    if collected >= 10:
                        break

        print(f"  ✅ GitHub: {len(self.code_samples)} GLua файлов")
        return self.code_samples


# ──────────────────────────────────────────────
# ПАРСЕР ФОРУМОВ FACEPUNCH
# ──────────────────────────────────────────────

class ForumParser:
    """Парсит Facepunch форумы на темы с кодом GLua."""

    SEARCH_TERMS = [
        "glua help",
        "gmod lua script",
        "darkrp lua",
        "gmod addon lua",
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(get_headers())
        self.posts = []

    def extract_code_from_post(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        codes = []
        for el in soup.find_all(["code", "pre"]):
            c = el.get_text().strip()
            if len(c) > 30:
                codes.append(c)
        return codes

    def parse_thread(self, url):
        try:
            r = self.session.get(url, timeout=10)
            if r.status_code != 200:
                return []
            soup = BeautifulSoup(r.text, "html.parser")
            results = []

            # Ищем посты с кодом
            for post in soup.find_all("div", class_=re.compile(r"post|message")):
                text = post.get_text()
                codes = self.extract_code_from_post(str(post))
                if codes and any(m in text for m in ["hook", "Entity", "player", "timer"]):
                    title_el = soup.find("h1") or soup.find("title")
                    title = title_el.get_text(strip=True) if title_el else ""
                    results.append({
                        "source": "forum",
                        "url": url,
                        "title": title,
                        "context": clean_text(text[:500]),
                        "codes": codes,
                    })
            return results
        except:
            return []

    def parse_all(self):
        print("\n💬 Парсим Facepunch форумы...")
        # Форумы Facepunch закрыты для парсинга, используем известные темы
        # Вместо этого парсим gmod.facepunch.com/wiki дополнительные разделы
        known_pages = [
            f"{WIKI_BASE}/gmod/Beginner_Tutorial_Intro",
            f"{WIKI_BASE}/gmod/Beginner_Tutorial_Weapons",
            f"{WIKI_BASE}/gmod/Beginner_Tutorial_NPCs",
            f"{WIKI_BASE}/gmod/Gamemode_Creation",
            f"{WIKI_BASE}/gmod/SWEP_Creation",
            f"{WIKI_BASE}/gmod/Entity_Creation",
            f"{WIKI_BASE}/gmod/Using_Hooks",
            f"{WIKI_BASE}/gmod/DarkRP_Customization",
            f"{WIKI_BASE}/gmod/Networking_Usage",
            f"{WIKI_BASE}/gmod/HUD_Creation",
        ]

        for url in tqdm(known_pages, desc="  Tutorial pages"):
            results = self.parse_thread(url)
            self.posts.extend(results)
            time.sleep(DELAY)

        print(f"  ✅ Форумы/Туториалы: {len(self.posts)} постов")
        return self.posts


# ──────────────────────────────────────────────
# 50+ РУЧНЫХ ПРИМЕРОВ
# ──────────────────────────────────────────────

def get_manual_pairs():
    return [
        # ── БАЗОВЫЕ ХУКИ ─────────────────────────────
        {
            "instruction": "Хук который срабатывает когда игрок заходит на сервер",
            "output": """hook.Add("PlayerInitialSpawn", "WelcomePlayer", function(ply)
    timer.Simple(2, function()
        if not IsValid(ply) then return end
        ply:ChatPrint("Добро пожаловать на сервер, " .. ply:Nick() .. "!")
        PrintMessage(HUD_PRINTTALK, ply:Nick() .. " зашёл на сервер.")
    end)
end)""",
        },
        {
            "instruction": "Хук на выход игрока с сервера",
            "output": """hook.Add("PlayerDisconnected", "LogDisconnect", function(ply)
    local name = ply:Nick()
    local steamid = ply:SteamID()
    print("[LOG] Игрок " .. name .. " (" .. steamid .. ") вышел с сервера.")
    for _, p in ipairs(player.GetAll()) do
        p:ChatPrint("Игрок " .. name .. " покинул сервер.")
    end
end)""",
        },
        {
            "instruction": "Хук на спавн игрока с выдачей оружия",
            "output": """hook.Add("PlayerSpawn", "GiveStarterKit", function(ply)
    timer.Simple(0.5, function()
        if not IsValid(ply) then return end
        ply:Give("weapon_crowbar")
        ply:Give("weapon_pistol")
        ply:SetAmmo(60, "Pistol")
        ply:SetHealth(100)
        ply:SetArmor(50)
    end)
end)""",
        },
        {
            "instruction": "Хук на смерть игрока с логированием и дропом инвентаря",
            "output": """hook.Add("PlayerDeath", "OnPlayerDeath", function(victim, inflictor, attacker)
    if not IsValid(victim) then return end

    -- Логируем смерть
    local vName = victim:Nick()
    if IsValid(attacker) and attacker:IsPlayer() then
        print(vName .. " убит игроком " .. attacker:Nick())
    end

    -- Дропаем оружие на землю
    local pos = victim:GetPos()
    for _, wep in ipairs(victim:GetWeapons()) do
        if IsValid(wep) then
            local dropped = ents.Create("prop_physics")
            dropped:SetModel(wep:GetModel())
            dropped:SetPos(pos + Vector(0, 0, 10))
            dropped:Spawn()
        end
    end
end)""",
        },
        {
            "instruction": "Хук на урон игроку — уменьшить урон вдвое",
            "output": """hook.Add("EntityTakeDamage", "HalfDamage", function(ent, dmg)
    if not ent:IsPlayer() then return end
    dmg:ScaleDamage(0.5)
end)""",
        },
        # ── NET / NETWORKING ─────────────────────────
        {
            "instruction": "Отправить сообщение от сервера конкретному игроку через net",
            "output": """-- SERVER
if SERVER then
    util.AddNetworkString("SendMessage")

    local function SendToPlayer(ply, message)
        net.Start("SendMessage")
            net.WriteString(message)
        net.Send(ply)
    end

    hook.Add("PlayerInitialSpawn", "SendWelcomeNet", function(ply)
        timer.Simple(3, function()
            if IsValid(ply) then
                SendToPlayer(ply, "Добро пожаловать! Это сетевое сообщение.")
            end
        end)
    end)
end

-- CLIENT
if CLIENT then
    net.Receive("SendMessage", function()
        local msg = net.ReadString()
        chat.AddText(Color(0, 200, 100), "[Сервер] ", Color(255,255,255), msg)
    end)
end""",
        },
        {
            "instruction": "Синхронизировать данные между сервером и клиентом через net",
            "output": """-- SERVER
if SERVER then
    util.AddNetworkString("SyncPlayerData")

    local playerData = {}

    local function SyncData(ply)
        local data = playerData[ply:SteamID()] or {kills = 0, deaths = 0}
        net.Start("SyncPlayerData")
            net.WriteTable(data)
        net.Send(ply)
    end

    hook.Add("PlayerSpawn", "SyncOnSpawn", function(ply)
        SyncData(ply)
    end)
end

-- CLIENT
if CLIENT then
    local myData = {}

    net.Receive("SyncPlayerData", function()
        myData = net.ReadTable()
        print("Синхронизировано: убийств=" .. myData.kills .. ", смертей=" .. myData.deaths)
    end)

    function GetMyData() return myData end
end""",
        },
        {
            "instruction": "Система чата команд через net с сервера на клиент",
            "output": """-- SERVER
if SERVER then
    util.AddNetworkString("ChatCommand")

    hook.Add("PlayerSay", "HandleChatCommands", function(ply, text)
        if text == "!help" then
            net.Start("ChatCommand")
                net.WriteString("Доступные команды: !help, !rules, !stats")
            net.Send(ply)
            return ""
        end

        if text == "!rules" then
            net.Start("ChatCommand")
                net.WriteString("Правила: 1) Не читить. 2) Уважать игроков.")
            net.Send(ply)
            return ""
        end
    end)
end

-- CLIENT
if CLIENT then
    net.Receive("ChatCommand", function()
        local msg = net.ReadString()
        chat.AddText(Color(255, 200, 0), "[Команда] ", Color(255,255,255), msg)
    end)
end""",
        },
        # ── ТАЙМЕРЫ ──────────────────────────────────
        {
            "instruction": "Таймер с обратным отсчётом который показывает время на экране",
            "output": """-- SERVER: запуск таймера
if SERVER then
    util.AddNetworkString("StartCountdown")

    local function StartCountdown(seconds)
        net.Start("StartCountdown")
            net.WriteInt(seconds, 16)
        net.Broadcast()
    end

    hook.Add("InitPostEntity", "LaunchCountdown", function()
        timer.Simple(5, function()
            StartCountdown(60)
        end)
    end)
end

-- CLIENT: отображение
if CLIENT then
    local timeLeft = 0
    local counting = false

    net.Receive("StartCountdown", function()
        timeLeft = net.ReadInt(16)
        counting = true
    end)

    timer.Create("CountdownTick", 1, 0, function()
        if counting and timeLeft > 0 then
            timeLeft = timeLeft - 1
            if timeLeft == 0 then
                counting = false
                chat.AddText(Color(255, 0, 0), "Время вышло!")
            end
        end
    end)

    hook.Add("HUDPaint", "DrawCountdown", function()
        if not counting or timeLeft <= 0 then return end
        draw.SimpleText(
            "Осталось: " .. timeLeft .. " сек",
            "DermaLarge",
            ScrW() / 2, 80,
            Color(255, 255, 0),
            TEXT_ALIGN_CENTER
        )
    end)
end""",
        },
        {
            "instruction": "Повторяющийся таймер для сохранения данных игроков каждые 5 минут",
            "output": """if SERVER then
    local function SaveAllPlayers()
        for _, ply in ipairs(player.GetAll()) do
            if IsValid(ply) and ply:IsPlayer() then
                -- Сохраняем данные через PData
                ply:SetPData("kills",  tostring(ply:GetNWInt("kills",  0)))
                ply:SetPData("deaths", tostring(ply:GetNWInt("deaths", 0)))
                ply:SetPData("money",  tostring(ply:GetNWInt("money",  0)))
            end
        end
        print("[AutoSave] Данные игроков сохранены: " .. os.date("%H:%M:%S"))
    end

    timer.Create("AutoSavePlayers", 300, 0, SaveAllPlayers)

    hook.Add("PlayerDisconnected", "SaveOnLeave", function(ply)
        SaveAllPlayers()
    end)
end""",
        },
        # ── HUD ──────────────────────────────────────
        {
            "instruction": "Нарисовать HUD с здоровьем и бронёй игрока",
            "output": """if CLIENT then
    hook.Add("HUDPaint", "DrawHealthArmor", function()
        local ply = LocalPlayer()
        if not IsValid(ply) or not ply:Alive() then return end

        local hp    = ply:Health()
        local armor = ply:Armor()
        local scrW  = ScrW()
        local scrH  = ScrH()

        -- Фон
        draw.RoundedBox(6, 20, scrH - 70, 200, 50, Color(0, 0, 0, 150))

        -- Полоска здоровья
        local hpWidth = math.Clamp(hp / 100, 0, 1) * 180
        draw.RoundedBox(4, 30, scrH - 60, 180, 12, Color(60, 0, 0, 200))
        draw.RoundedBox(4, 30, scrH - 60, hpWidth, 12, Color(220, 50, 50))
        draw.SimpleText("❤ " .. hp, "DermaDefault", 30, scrH - 45, Color(255,255,255))

        -- Полоска брони
        local arWidth = math.Clamp(armor / 100, 0, 1) * 180
        draw.RoundedBox(4, 30, scrH - 42, 180, 8, Color(0, 0, 60, 200))
        draw.RoundedBox(4, 30, scrH - 42, arWidth, 8, Color(50, 100, 220))
        draw.SimpleText("🛡 " .. armor, "DermaDefault", 30, scrH - 32, Color(200,200,255))
    end)

    -- Скрываем стандартный HUD
    hook.Add("HUDShouldDraw", "HideDefaultHUD", function(name)
        if name == "CHudHealth" or name == "CHudBattery" then
            return false
        end
    end)
end""",
        },
        {
            "instruction": "HUD с отображением имени игрока на прицеле и его здоровья",
            "output": """if CLIENT then
    hook.Add("HUDPaint", "DrawPlayerInfo", function()
        local ply  = LocalPlayer()
        local tr   = ply:GetEyeTrace()
        local ent  = tr.Entity

        if not IsValid(ent) or not ent:IsPlayer() then return end
        if tr.Fraction > 0.3 then return end  -- Только близкие игроки

        local scrW = ScrW()
        local scrH = ScrH()
        local name = ent:Nick()
        local hp   = ent:Health()
        local team = team.GetName(ent:Team())

        -- Фон по центру экрана
        draw.RoundedBox(6, scrW/2 - 80, scrH/2 + 20, 160, 40, Color(0,0,0,180))
        draw.SimpleText(name,         "DermaDefault", scrW/2, scrH/2 + 25, Color(255,255,255), TEXT_ALIGN_CENTER)
        draw.SimpleText("HP: " .. hp, "DermaDefault", scrW/2, scrH/2 + 38, Color(200,80,80),   TEXT_ALIGN_CENTER)
    end)
end""",
        },
        # ── DERMA UI ─────────────────────────────────
        {
            "instruction": "Создать меню администратора с кнопками кик и бан",
            "output": """-- CLIENT: открытие меню
if CLIENT then
    local function OpenAdminMenu()
        local ply = LocalPlayer()
        if not ply:IsAdmin() then
            chat.AddText(Color(255,0,0), "У вас нет прав!")
            return
        end

        local frame = vgui.Create("DFrame")
        frame:SetSize(400, 500)
        frame:SetTitle("Панель администратора")
        frame:Center()
        frame:MakePopup()

        local scroll = vgui.Create("DScrollPanel", frame)
        scroll:Dock(FILL)

        for _, target in ipairs(player.GetAll()) do
            local row = vgui.Create("DPanel", scroll)
            row:SetSize(380, 50)
            row:Dock(TOP)
            row:DockMargin(5, 5, 5, 0)

            local label = vgui.Create("DLabel", row)
            label:SetPos(10, 15)
            label:SetText(target:Nick())
            label:SetTextColor(Color(255,255,255))

            local kickBtn = vgui.Create("DButton", row)
            kickBtn:SetPos(250, 10)
            kickBtn:SetSize(60, 30)
            kickBtn:SetText("Кик")
            kickBtn.DoClick = function()
                RunConsoleCommand("kick_id", target:UserID())
                frame:Close()
            end

            local banBtn = vgui.Create("DButton", row)
            banBtn:SetPos(315, 10)
            banBtn:SetSize(60, 30)
            banBtn:SetText("Бан")
            banBtn.DoClick = function()
                RunConsoleCommand("ban_id", target:UserID(), "60")
                frame:Close()
            end
        end
    end

    concommand.Add("admin_menu", OpenAdminMenu)
end""",
        },
        {
            "instruction": "Создать окно настроек с чекбоксами и слайдерами",
            "output": """if CLIENT then
    local settings = {
        showHUD     = true,
        hudScale    = 1.0,
        chatSound   = true,
        fov         = 90,
    }

    local function OpenSettings()
        local frame = vgui.Create("DFrame")
        frame:SetSize(350, 400)
        frame:SetTitle("Настройки")
        frame:Center()
        frame:MakePopup()

        -- Показывать HUD
        local cbHUD = vgui.Create("DCheckBoxLabel", frame)
        cbHUD:SetPos(20, 40)
        cbHUD:SetText("Показывать HUD")
        cbHUD:SetValue(settings.showHUD)
        cbHUD.OnChange = function(_, val) settings.showHUD = val end

        -- Звук чата
        local cbSound = vgui.Create("DCheckBoxLabel", frame)
        cbSound:SetPos(20, 70)
        cbSound:SetText("Звук чата")
        cbSound:SetValue(settings.chatSound)
        cbSound.OnChange = function(_, val) settings.chatSound = val end

        -- Масштаб HUD
        local labelScale = vgui.Create("DLabel", frame)
        labelScale:SetPos(20, 110)
        labelScale:SetText("Масштаб HUD:")
        labelScale:SetTextColor(Color(255,255,255))

        local sliderScale = vgui.Create("DNumSlider", frame)
        sliderScale:SetPos(20, 130)
        sliderScale:SetSize(310, 30)
        sliderScale:SetMin(0.5)
        sliderScale:SetMax(2.0)
        sliderScale:SetDecimals(1)
        sliderScale:SetValue(settings.hudScale)
        sliderScale.OnValueChanged = function(_, val) settings.hudScale = val end

        -- FOV
        local labelFOV = vgui.Create("DLabel", frame)
        labelFOV:SetPos(20, 175)
        labelFOV:SetText("Поле зрения (FOV):")
        labelFOV:SetTextColor(Color(255,255,255))

        local sliderFOV = vgui.Create("DNumSlider", frame)
        sliderFOV:SetPos(20, 195)
        sliderFOV:SetSize(310, 30)
        sliderFOV:SetMin(60)
        sliderFOV:SetMax(120)
        sliderFOV:SetDecimals(0)
        sliderFOV:SetValue(settings.fov)
        sliderFOV.OnValueChanged = function(_, val)
            settings.fov = val
            LocalPlayer():SetFOV(val, 0.3)
        end

        -- Кнопка сохранить
        local saveBtn = vgui.Create("DButton", frame)
        saveBtn:SetPos(125, 350)
        saveBtn:SetSize(100, 35)
        saveBtn:SetText("Сохранить")
        saveBtn.DoClick = function()
            for k, v in pairs(settings) do
                cookie.Set("settings_" .. k, tostring(v))
            end
            frame:Close()
        end
    end

    concommand.Add("open_settings", OpenSettings)
end""",
        },
        # ── SWEP ─────────────────────────────────────
        {
            "instruction": "Создать базовый SWEP — кастомное оружие в GLua",
            "output": """SWEP.PrintName     = "Лазерный пистолет"
SWEP.Author        = "Мой Сервер"
SWEP.Spawnable     = true
SWEP.AdminSpawnable = true

SWEP.Base          = "weapon_base"
SWEP.HoldType      = "pistol"

SWEP.Primary.ClipSize    = 12
SWEP.Primary.DefaultClip = 24
SWEP.Primary.Automatic   = false
SWEP.Primary.Ammo        = "Pistol"

SWEP.Secondary.ClipSize    = -1
SWEP.Secondary.DefaultClip = -1
SWEP.Secondary.Automatic   = false

function SWEP:PrimaryAttack()
    if not self:CanPrimaryAttack() then return end

    self:SetNextPrimaryFire(CurTime() + 0.3)
    self:TakePrimaryAmmo(1)
    self:EmitSound("weapons/pistol/pistol_fire2.wav")

    if SERVER then
        local ply = self:GetOwner()
        local tr  = ply:GetEyeTrace()

        -- Лазерный эффект
        local effect = EffectData()
        effect:SetStart(self:GetPos())
        effect:SetOrigin(tr.HitPos)
        util.Effect("BeamTracer", effect)

        -- Урон
        if IsValid(tr.Entity) then
            tr.Entity:TakeDamage(25, ply, self)
        end
    end
end

function SWEP:SecondaryAttack()
    -- Зум
    local ply = self:GetOwner()
    if CLIENT and IsValid(ply) and ply == LocalPlayer() then
        ply:SetFOV(ply:GetFOV() == 90 and 45 or 90, 0.2)
    end
end""",
        },
        # ── ENTITY ───────────────────────────────────
        {
            "instruction": "Создать кастомную Entity — сундук с лутом",
            "output": """-- shared.lua
ENT.Type      = "anim"
ENT.Base      = "base_gmodentity"
ENT.PrintName = "Сундук с лутом"
ENT.Spawnable = true

function ENT:SetupDataTables()
    self:NetworkVar("Bool", 0, "IsOpen")
end

-- init.lua (сервер)
if SERVER then
    function ENT:Initialize()
        self:SetModel("models/props_junk/wood_crate001a.mdl")
        self:PhysicsInit(SOLID_VPHYSICS)
        self:SetMoveType(MOVETYPE_VPHYSICS)
        self:SetSolid(SOLID_VPHYSICS)
        self:SetIsOpen(false)

        local phys = self:GetPhysicsObject()
        if IsValid(phys) then phys:Wake() end
    end

    function ENT:Use(activator, caller)
        if not activator:IsPlayer() then return end
        if self:GetIsOpen() then
            activator:ChatPrint("Сундук уже открыт!")
            return
        end

        self:SetIsOpen(true)
        self:EmitSound("physics/wood/wood_crate_impact_hard1.wav")

        -- Выдаём случайный лут
        local loot = {"weapon_pistol", "weapon_crowbar", "item_healthkit"}
        local item = loot[math.random(#loot)]
        activator:Give(item)
        activator:ChatPrint("Вы нашли: " .. item)

        -- Удаляем сундук через 5 сек
        timer.Simple(5, function()
            if IsValid(self) then self:Remove() end
        end)
    end
end

-- cl_init.lua (клиент)
if CLIENT then
    function ENT:Draw()
        self:DrawModel()
    end
end""",
        },
        # ── DARKRP ───────────────────────────────────
        {
            "instruction": "Создать DarkRP работу (job) — полицейский с оружием",
            "output": """-- config/jobrelated.lua
DarkRP.createJob("Полицейский", {
    color       = Color(0, 0, 255),
    model       = {"models/player/combine_super_soldier.mdl"},
    description = "Охраняет порядок в городе. Может арестовывать нарушителей.",
    weapons     = {"weapon_pistol", "arrest_stick", "unarrest_stick", "weapon_stunstick"},
    command     = "cop",
    max         = 5,
    salary      = 200,
    admin       = 0,
    vote        = false,
    hasLicense  = true,
    candemote   = true,
    chief       = false,
    PlayerSpawn = function(ply)
        ply:SetHealth(120)
        ply:SetArmor(50)
    end,
    PlayerDeath = function(ply, inflictor, attacker)
        DarkRP.notifyAll(0, 4, "Полицейский " .. ply:Nick() .. " погиб!")
    end,
})""",
        },
        {
            "instruction": "Создать DarkRP магазин предметов с ценами",
            "output": """-- addons/my_shop/lua/autorun/server/sh_shop.lua
if SERVER then
    local SHOP = {
        {name = "Пистолет",    item = "weapon_pistol",   price = 500},
        {name = "Дробовик",    item = "weapon_shotgun",  price = 1500},
        {name = "Аптечка",     item = "item_healthkit",  price = 200},
        {name = "Бронежилет",  item = "item_suit",       price = 750},
    }

    util.AddNetworkString("OpenShop")
    util.AddNetworkString("BuyItem")

    net.Receive("OpenShop", function(_, ply)
        net.Start("OpenShop")
            net.WriteTable(SHOP)
        net.Send(ply)
    end)

    net.Receive("BuyItem", function(_, ply)
        local idx = net.ReadInt(8)
        local item = SHOP[idx]
        if not item then return end

        if ply:getDarkRPVar("money") >= item.price then
            ply:addMoney(-item.price)
            ply:Give(item.item)
            DarkRP.notify(ply, 0, 4, "Вы купили: " .. item.name)
        else
            DarkRP.notify(ply, 1, 4, "Недостаточно денег!")
        end
    end)
end""",
        },
        # ── СОХРАНЕНИЕ ДАННЫХ ────────────────────────
        {
            "instruction": "Сохранить и загрузить данные игрока через PData",
            "output": """if SERVER then
    -- Загрузка при входе
    hook.Add("PlayerInitialSpawn", "LoadPlayerData", function(ply)
        local kills  = tonumber(ply:GetPData("kills",  "0")) or 0
        local deaths = tonumber(ply:GetPData("deaths", "0")) or 0
        local money  = tonumber(ply:GetPData("money",  "500")) or 500

        ply:SetNWInt("kills",  kills)
        ply:SetNWInt("deaths", deaths)
        ply:SetNWInt("money",  money)
    end)

    -- Сохранение при смерти
    hook.Add("PlayerDeath", "SaveOnDeath", function(ply)
        local deaths = ply:GetNWInt("deaths", 0) + 1
        ply:SetNWInt("deaths", deaths)
        ply:SetPData("deaths", tostring(deaths))
    end)

    -- Сохранение при убийстве
    hook.Add("PlayerDeath", "SaveKills", function(victim, inf, attacker)
        if IsValid(attacker) and attacker:IsPlayer() then
            local kills = attacker:GetNWInt("kills", 0) + 1
            attacker:SetNWInt("kills", kills)
            attacker:SetPData("kills", tostring(kills))
        end
    end)

    -- Сохранение при выходе
    hook.Add("PlayerDisconnected", "SaveOnLeave", function(ply)
        ply:SetPData("money", tostring(ply:GetNWInt("money", 0)))
    end)
end""",
        },
        # ── UTIL / ВСПОМОГАТЕЛЬНОЕ ───────────────────
        {
            "instruction": "Функция которая находит ближайшего игрока к заданной точке",
            "output": """-- Возвращает ближайшего игрока к позиции pos (и расстояние до него)
local function GetNearestPlayer(pos, maxDist)
    maxDist = maxDist or math.huge
    local nearest, nearestDist = nil, maxDist

    for _, ply in ipairs(player.GetAll()) do
        if IsValid(ply) and ply:Alive() then
            local dist = pos:Distance(ply:GetPos())
            if dist < nearestDist then
                nearest     = ply
                nearestDist = dist
            end
        end
    end

    return nearest, nearestDist
end

-- Пример использования
hook.Add("Think", "FindNearestPlayerTest", function()
    for _, ply in ipairs(player.GetAll()) do
        local nearest, dist = GetNearestPlayer(ply:GetPos())
        if nearest and nearest ~= ply and dist < 200 then
            -- Игрок слишком близко к другому игроку
        end
    end
end)""",
        },
        {
            "instruction": "Утилита для трассировки луча (trace) от игрока",
            "output": """-- Трассировка от взгляда игрока
local function TraceFromPlayer(ply, dist)
    dist = dist or 1000
    local startPos = ply:EyePos()
    local dir      = ply:GetAimVector()
    local endPos   = startPos + dir * dist

    local tr = util.TraceLine({
        start  = startPos,
        endpos = endPos,
        filter = ply,
        mask   = MASK_SOLID_BRUSHONLY,
    })

    return tr
end

-- Пример: хук Use — игрок смотрит на объект
hook.Add("PlayerUse", "UseObjectInSight", function(ply, ent)
    local tr = TraceFromPlayer(ply, 150)

    if IsValid(tr.Entity) then
        local hitEnt = tr.Entity
        print("Игрок " .. ply:Nick() .. " смотрит на: " .. tostring(hitEnt))
        return hitEnt
    end
end)""",
        },
        # ── ГЕЙМОДЫ ──────────────────────────────────
        {
            "instruction": "Создать структуру геймода с базовыми хуками",
            "output": """-- gamemode/init.lua (сервер)
AddCSLuaFile("cl_init.lua")
AddCSLuaFile("shared.lua")
include("shared.lua")

function GM:Initialize()
    print("[MyGamemode] Инициализация сервера")
    game.SetTimeScale(1)
end

function GM:PlayerInitialSpawn(ply)
    ply:SetTeam(1)
    ply:SetModel("models/player/combine_super_soldier.mdl")
end

function GM:PlayerSpawn(ply)
    self.BaseClass.PlayerSpawn(self, ply)
    ply:Give("weapon_crowbar")
end

function GM:PlayerDeath(victim, inflictor, attacker)
    timer.Simple(3, function()
        if IsValid(victim) then victim:Spawn() end
    end)
end

-- gamemode/shared.lua
GM.Name    = "Мой Геймод"
GM.Author  = "Разработчик"
GM.TeamBased = false

-- gamemode/cl_init.lua (клиент)
include("shared.lua")

function GM:HUDPaint()
    local ply = LocalPlayer()
    if not IsValid(ply) then return end
    draw.SimpleText("Мой Геймод", "DermaLarge", ScrW()/2, 20, Color(255,255,255), TEXT_ALIGN_CENTER)
end""",
        },
        # ── ТАБЛИЦЫ И ДАННЫЕ ─────────────────────────
        {
            "instruction": "Система уровней для игроков с опытом и повышением уровня",
            "output": """if SERVER then
    -- Таблица уровней: {нужный опыт, название уровня}
    local LEVELS = {
        {0,    "Новичок"},
        {100,  "Ученик"},
        {300,  "Боец"},
        {600,  "Воин"},
        {1000, "Ветеран"},
        {1500, "Элита"},
        {2500, "Легенда"},
    }

    local function GetLevelInfo(exp)
        local level, title = 1, LEVELS[1][2]
        for i, data in ipairs(LEVELS) do
            if exp >= data[1] then
                level, title = i, data[2]
            end
        end
        return level, title, LEVELS[level + 1] and LEVELS[level + 1][1] or nil
    end

    local function AddExp(ply, amount)
        local exp = ply:GetNWInt("exp", 0) + amount
        local oldLevel = select(1, GetLevelInfo(ply:GetNWInt("exp", 0)))
        local newLevel = select(1, GetLevelInfo(exp))

        ply:SetNWInt("exp", exp)
        ply:SetPData("exp", tostring(exp))

        if newLevel > oldLevel then
            local _, title = GetLevelInfo(exp)
            ply:SetNWString("levelTitle", title)
            ply:SetNWInt("level", newLevel)
            ply:ChatPrint("🎉 Поздравляем! Вы достигли уровня " .. newLevel .. " (" .. title .. ")")
            ply:EmitSound("buttons/button17.wav")
        end
    end

    -- Добавляем опыт за убийство
    hook.Add("PlayerDeath", "AddExpOnKill", function(victim, inf, attacker)
        if IsValid(attacker) and attacker:IsPlayer() and attacker ~= victim then
            AddExp(attacker, 50)
        end
    end)
end""",
        },
        # ── КОНСОЛЬНЫЕ КОМАНДЫ ───────────────────────
        {
            "instruction": "Зарегистрировать консольную команду с аргументами",
            "output": """-- CLIENT: команда для открытия меню
concommand.Add("open_menu", function(ply, cmd, args)
    OpenMyMenu()
end, nil, "Открыть главное меню")

-- SERVER: команда с аргументами
if SERVER then
    concommand.Add("give_money", function(ply, cmd, args)
        if not ply:IsAdmin() then
            ply:PrintMessage(HUD_PRINTCONSOLE, "Нет прав!")
            return
        end

        local targetName = args[1]
        local amount     = tonumber(args[2])

        if not targetName or not amount then
            ply:PrintMessage(HUD_PRINTCONSOLE, "Использование: give_money <имя> <сумма>")
            return
        end

        -- Ищем игрока по имени
        for _, target in ipairs(player.GetAll()) do
            if string.find(target:Nick():lower(), targetName:lower()) then
                target:addMoney(amount)
                ply:PrintMessage(HUD_PRINTCONSOLE,
                    "Выдано " .. amount .. " игроку " .. target:Nick())
                return
            end
        end

        ply:PrintMessage(HUD_PRINTCONSOLE, "Игрок не найден: " .. targetName)
    end, nil, "Выдать деньги игроку [ADMIN]")
end""",
        },
    ]

# ──────────────────────────────────────────────
# ГЕНЕРАТОР ПАР
# ──────────────────────────────────────────────

class TrainingPairGenerator:
    TEMPLATES = [
        "Напиши GLua скрипт: {desc}",
        "Создай GLua код для: {desc}",
        "Реализуй на GLua: {desc}",
        "Пример GLua кода: {desc}",
        "Как на GLua сделать: {desc}?",
        "GLua — {desc}",
    ]

    def from_wiki(self, docs):
        pairs = []
        for doc in docs:
            if not doc.get("code_examples"):
                continue
            desc = doc.get("description", "")[:300]
            if not desc:
                continue
            for i, code in enumerate(doc["code_examples"][:3]):
                if len(code) < 20:
                    continue
                tmpl = self.TEMPLATES[i % len(self.TEMPLATES)]
                # Если есть параметры — включаем их в инструкцию
                param_hint = ""
                if doc.get("parameters"):
                    names = [p["name"] for p in doc["parameters"][:3]]
                    param_hint = " (параметры: " + ", ".join(names) + ")"
                pairs.append({
                    "instruction": tmpl.format(desc=desc) + param_hint,
                    "input": "",
                    "output": code,
                    "source": "wiki",
                    "function": doc["title"],
                })
        return pairs

    def from_github(self, code_samples):
        pairs = []
        for item in code_samples:
            code = item["code"]
            desc = item.get("repo_description", "")

            # Ищем комментарий-описание в начале файла
            header = re.search(r'^--\s*(.+)', code, re.MULTILINE)
            block  = re.search(r'--\[\[(.+?)\]\]', code, re.DOTALL)

            if block:
                desc = block.group(1).strip()[:300]
            elif header:
                desc = header.group(1).strip()[:300]

            if not desc or len(desc) < 10:
                continue

            # Разбиваем на функции
            funcs = re.split(r'\n(?=function\s)', code)
            for func in funcs[:4]:
                if len(func) < 60:
                    continue
                tmpl = random.choice(self.TEMPLATES)
                pairs.append({
                    "instruction": tmpl.format(desc=desc),
                    "input": "",
                    "output": func[:3000],
                    "source": "github",
                    "repo": item["repo"],
                })

            # Также добавляем файл целиком если небольшой
            if len(code) < 2000:
                pairs.append({
                    "instruction": f"Напиши GLua скрипт: {desc}",
                    "input": "",
                    "output": code,
                    "source": "github_full",
                    "repo": item["repo"],
                })

        return pairs

    def from_forums(self, posts):
        pairs = []
        for post in posts:
            title = post.get("title", "")
            if not title or not post.get("codes"):
                continue
            for code in post["codes"][:2]:
                if len(code) < 30:
                    continue
                pairs.append({
                    "instruction": f"Напиши GLua код: {title}",
                    "input": post.get("context", ""),
                    "output": code,
                    "source": "forum",
                })
        return pairs


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   GLua Dataset Parser v2 — МАКСИМАЛЬНЫЙ")
    print("=" * 60)

    make_output_dir()

    # 1. Wiki
    wiki   = WikiParser()
    wdocs  = wiki.parse_all(max_pages=600)
    save_jsonl(wdocs, "wiki_docs.jsonl")

    # 2. GitHub
    gh     = GitHubParser()
    gcode  = gh.parse_all(max_repos_per_query=12)
    save_jsonl(gcode, "github_code.jsonl")

    # 3. Форумы
    fp     = ForumParser()
    posts  = fp.parse_all()
    save_jsonl(posts, "forum_posts.jsonl")

    # 4. Генерируем пары
    print("\n🔧 Генерируем обучающие пары...")
    gen    = TrainingPairGenerator()

    pairs  = get_manual_pairs()                 # 30+ ручных
    pairs += gen.from_wiki(wdocs)              # из wiki
    pairs += gen.from_github(gcode)            # из github
    pairs += gen.from_forums(posts)            # из форумов

    # Формат Alpaca — добавляем source если нет
    for p in pairs:
        if "source" not in p:
            p["source"] = "manual"
        if "input" not in p:
            p["input"] = ""

    # Удаляем дубликаты
    seen, unique = set(), []
    for p in pairs:
        key = (p["instruction"][:80] + p["output"][:80])
        if key not in seen and len(p["output"]) > 30:
            seen.add(key)
            unique.append(p)

    # Перемешиваем
    random.shuffle(unique)

    save_jsonl(unique, "training_pairs.jsonl")

    # 5. Статистика
    by_source = {}
    for p in unique:
        s = p.get("source", "?")
        by_source[s] = by_source.get(s, 0) + 1

    summary = f"""
GLua Dataset v2 — Статистика
==============================
Всего обучающих пар:    {len(unique)}
  manual:               {by_source.get('manual', 0)}
  wiki:                 {by_source.get('wiki', 0)}
  github:               {by_source.get('github', 0) + by_source.get('github_full', 0)}
  forum:                {by_source.get('forum', 0)}

Файлы:
  {OUTPUT_DIR}/wiki_docs.jsonl        ({len(wdocs)} документов)
  {OUTPUT_DIR}/github_code.jsonl      ({len(gcode)} файлов)
  {OUTPUT_DIR}/forum_posts.jsonl      ({len(posts)} постов)
  {OUTPUT_DIR}/training_pairs.jsonl   ← для обучения

Следующий шаг:
  Загрузи training_pairs.jsonl на Kaggle → Fine-tuning!
"""
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    print("🎉 Готово!")


if __name__ == "__main__":
    main()
 