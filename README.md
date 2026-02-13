# HWPX Skill

품질 검증 중심 HWPX 스킬입니다.  
핵심 파일은 `SKILL.md`, `scripts/hwpx_tool.py`, `scripts/render_hwpx.py` 입니다.

## GitHub 링크만으로 설치 (권장)

아래 한 줄로 설치할 수 있습니다.

```bash
python3 "$CODEX_HOME/skills/.system/skill-installer/scripts/install-skill-from-github.py" \
  --url "https://github.com/koolerjaebee/hwpx-skill/tree/main"
```

설치 후 Codex를 재시작하세요.

## 수동 설치 (fallback)

`skill-installer`를 쓰기 어려운 환경이면 수동으로 설치할 수 있습니다.

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
git clone https://github.com/koolerjaebee/hwpx-skill.git \
  "${CODEX_HOME:-$HOME/.codex}/skills/hwpx"
```

설치 후 Codex를 재시작하세요.

## 설치 확인

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" \
  "${CODEX_HOME:-$HOME/.codex}/skills/hwpx"
```

정상 설치면 `Skill is valid!`가 출력됩니다.
